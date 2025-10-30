
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal SVD-LoRA Distillation (Dense + MoE) — v1.1

What's new in v1.1
------------------
- **Fix**: When the teacher is MoE (experts under `mlp.experts`) but the student
  is dense (`mlp.down_proj/gate_proj/up_proj`), MLP LoRA wasn't emitted.
  v1.1 adds **MoE->Dense MLP synthesis**:
    * Reads per-layer router (`mlp.gate.weight`) to score experts.
    * Blends top-K experts (softmax over L2 norms) across teacher floor/ceil
      layers, then projects to student's shape and extracts LoRA.
    * Fallbacks to uniform averaging if router is missing.

- **Compatibility**: Keeps prior Mixtral-style `block_sparse_moe.experts` support,
  and additionally recognizes Qwen-style `mlp.experts` naming.

Author: Adapted from a MoE-only baseline provided by the user.
License: MIT
"""
import os
import re
import json
import math
import time
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm

try:
    from safetensors.torch import load_file, save_file, safe_open
except Exception as e:
    raise RuntimeError("safetensors is required. pip install safetensors") from e

try:
    import faiss
except Exception:
    faiss = None

# =================================================================================
#                                  CLI ARGUMENTS
# =================================================================================
def build_argparser():
    p = argparse.ArgumentParser(description="Universal SVD-LoRA Distillation (Dense + MoE)")
    # IO
    p.add_argument("--teacher", type=str, required=True)
    p.add_argument("--student", type=str, required=True)
    p.add_argument("--out_lora", type=str, required=True)
    p.add_argument("--out_adapter", type=str, default=None)
    # Performance
    p.add_argument("--num_gpus", type=int, default=None)
    p.add_argument("--micro_bs", type=int, default=16)

    # LoRA / ranks
    p.add_argument("--rank_default", type=int, default=None)
    p.add_argument("--rank_attn", type=int, default=None)
    p.add_argument("--rank_mlp", type=int, default=None)
    p.add_argument("--rank_moe", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    # Filtering
    p.add_argument("--include", type=str, default=None)
    p.add_argument("--exclude", type=str, default=None)
    p.add_argument("--include_embed_lm_head", action="store_true")
    # MoE controls
    p.add_argument("--kmeans_iter", type=int, default=25)
    p.add_argument("--max_teachers_to_blend", type=int, default=160)
    p.add_argument("--fallback_uniform_moe", action="store_true")
    # Layer mapping
    p.add_argument("--map_schedule", type=str, default="sigmoid", choices=["sigmoid","linear"])
    p.add_argument("--sigmoid_k", type=float, default=0.15)
    # Advanced
    p.add_argument("--seed", type=int, default=1234)
    return p

# =================================================================================
#                               ARCH DETECTION / UTILS
# =================================================================================

INDEX_PATTERNS = [
    ("layers", re.compile(r"^(?P<prefix>.*?)(layers)\.(?P<idx>\d+)\.(?P<rest>.+)$")),
    ("h", re.compile(r"^(?P<prefix>.*?)(h)\.(?P<idx>\d+)\.(?P<rest>.+)$")),
    ("blocks", re.compile(r"^(?P<prefix>.*?)(blocks)\.(?P<idx>\d+)\.(?P<rest>.+)$")),
    ("encoder.layer", re.compile(r"^(?P<prefix>.*?)(encoder\.layer)\.(?P<idx>\d+)\.(?P<rest>.+)$")),
    ("decoder.layer", re.compile(r"^(?P<prefix>.*?)(decoder\.layer)\.(?P<idx>\d+)\.(?P<rest>.+)$")),
    ("encoder.block", re.compile(r"^(?P<prefix>.*?)(encoder\.block)\.(?P<idx>\d+)\.(?P<rest>.+)$")),
    ("decoder.block", re.compile(r"^(?P<prefix>.*?)(decoder\.block)\.(?P<idx>\d+)\.(?P<rest>.+)$")),
]

NORM_TOKENS = ("norm", "layer_norm", "ln_", "layernorm", "rms_norm", "input_layernorm", "post_attention_layernorm")
EMBED_TOKENS = ("embed", "wte", "word_embeddings", "position_embeddings", "lm_head", "lm_head.weight")
SKIP_SUFFIXES = (".bias",)

ATTN_HINTS = ("attn", "attention", "self_attn", "self_attention", "q_proj", "k_proj", "v_proj", "o_proj", "query_key_value", "c_attn", "Wqkv")
MLP_HINTS = ("mlp", "feed_forward", "ffn", "dense_h_to_4h", "dense_4h_to_h", "gate_proj", "up_proj", "down_proj", "c_fc", "w1", "w2", "w3")

def seed_all(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_index_map(folder: str):
    idx_path = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing index file: {idx_path}")
    return load_json(idx_path)["weight_map"]

def scan_layers(weight_keys):
    layers = defaultdict(set)
    for k in weight_keys:
        for name, rx in INDEX_PATTERNS:
            m = rx.match(k)
            if m:
                layers[name].add(int(m.group("idx"))); break
    return {name: sorted(list(v)) for name, v in layers.items()}

def get_layer_count_for_key(key: str, teacher_layers_map: dict, student_layers_map: dict):
    for name, rx in INDEX_PATTERNS:
        if rx.match(key):
            s_layers = len(student_layers_map.get(name, []))
            t_layers = len(teacher_layers_map.get(name, []))
            if s_layers > 0 and t_layers > 0:
                return name, s_layers, t_layers
    if student_layers_map:
        best = max(student_layers_map.items(), key=lambda kv: len(kv[1]))[0]
        return best, len(student_layers_map.get(best, [])), len(teacher_layers_map.get(best, []))
    return None, 0, 0

def split_key(key: str):
    for name, rx in INDEX_PATTERNS:
        m = rx.match(key)
        if m: return m.group("prefix"), name, int(m.group("idx")), m.group("rest")
    return None

def build_same_suffix_key(student_key: str, student_prefix: str, teacher_prefix: str):
    tail = student_key[len(student_prefix):] if student_key.startswith(student_prefix) else student_key
    return teacher_prefix + tail

def find_prefix(weight_keys, token_example="layers"):
    counts = defaultdict(int)
    for k in weight_keys:
        for name, rx in INDEX_PATTERNS:
            m = rx.match(k)
            if m and name == token_example:
                counts[m.group("prefix")] += 1
    if not counts:
        for k in weight_keys:
            i = k.find(".")
            if i > 0: counts[k[:i+1]] += 1
    if counts: return max(counts.items(), key=lambda kv: kv[1])[0]
    return ""

def is_linear_2d_shape(t):
    return (t is not None) and (t.dim() == 2)



# --- Helper: normalize LoRA key names to PEFT-style ---
def to_lora_key(param_key: str, student_prefix_guess: str, base_prefix: str = "base_model.model.model.") -> str:
    """
    Turn e.g. 'model.layers.0.self_attn.k_proj.weight' into
    'base_model.model.model.layers.0.self_attn.k_proj'.
    1) strip trailing '.weight'
    2) replace leading student prefix with base_prefix
    """
    k = param_key
    if k.endswith(".weight"):
        k = k[:-len(".weight")]
    # If already has target base prefix, leave as-is
    if k.startswith(base_prefix):
        return k
    # Replace the auto-detected student prefix with the requested base prefix
    if k.startswith(student_prefix_guess):
        k = k[len(student_prefix_guess):]
    return base_prefix + k

# =================================================================================
#                                  LOADING UTILS
# =================================================================================

def get_tensors_from_shards(keys_to_find, model_folder, weight_map, device="cpu"):
    shards_to_load = defaultdict(list)
    for key in keys_to_find:
        if key in weight_map:
            shards_to_load[weight_map[key]].append(key)
    tensors = {}
    for shard_file, keys_in_shard in shards_to_load.items():
        shard_path = os.path.join(model_folder, shard_file)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in keys_in_shard:
                try: tensors[key] = f.get_tensor(key)
                except Exception: continue
    return tensors

def get_scaled_fp8_tensors(keys_to_find, model_folder, weight_map, device="cpu"):
    shards_to_load = defaultdict(list)
    keys_with_scales = list(set(keys_to_find + [f"{key}.scales" for key in keys_to_find]))
    for key in keys_with_scales:
        if key in weight_map:
            shards_to_load[weight_map[key]].append(key)
    tensors = {}
    for shard_file, keys_in_shard in shards_to_load.items():
        shard_path = os.path.join(model_folder, shard_file)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in keys_in_shard:
                try: tensors[key] = f.get_tensor(key)
                except Exception: continue
    deq = {}
    for key in keys_to_find:
        scale_key = f"{key}.scales"
        if key in tensors and scale_key in tensors:
            v = tensors[key].float() * tensors[scale_key].float()
            deq[key] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        elif key in tensors:
            deq[key] = torch.nan_to_num(tensors[key].float(), nan=0.0, posinf=0.0, neginf=0.0)
    return deq

# =================================================================================
#                               LINEAR ALGEBRA UTILS
# =================================================================================

def slerp(t1, t2, weight, epsilon=1e-7):
    original_dtype = t1.dtype
    t1_flat = t1.flatten().float(); t2_flat = t2.flatten().float()
    t1_norm_val = torch.linalg.vector_norm(t1_flat); t2_norm_val = torch.linalg.vector_norm(t2_flat)
    t1_norm = t1_flat / (t1_norm_val + epsilon); t2_norm = t2_flat / (t2_norm_val + epsilon)
    dot = torch.clamp(torch.dot(t1_norm, t2_norm), -1.0, 1.0); theta = torch.acos(dot)
    if theta < 1e-4: return (t1 * (1 - weight) + t2 * weight).reshape(t1.shape)
    sin_theta = torch.sin(theta)
    s1 = torch.sin((1 - weight) * theta) / sin_theta; s2 = torch.sin(weight * theta) / sin_theta
    interpolated_norm_flat = s1 * t1_norm + s2 * t2_norm
    interpolated_original_norm = t1_norm_val * (1 - weight) + t2_norm_val * weight
    return (interpolated_norm_flat * interpolated_original_norm).reshape(t1.shape).to(original_dtype)

def apply_dare_ties(diff_tensor, drop_rate=0.8):
    if diff_tensor.dim() != 2: return diff_tensor
    safe = torch.nan_to_num(diff_tensor, nan=0.0, posinf=0.0, neginf=0.0).float()
    mag = torch.abs(safe)
    if torch.all(mag == 0): return diff_tensor
    flat = mag.flatten(); k = int(flat.numel() * (1 - drop_rate))
    if k == 0: return torch.zeros_like(diff_tensor)
    threshold = torch.kthvalue(flat, flat.numel() - k).values
    mask = mag >= threshold
    pruned = safe * mask
    on = torch.linalg.vector_norm(safe); pn = torch.linalg.vector_norm(pruned)
    if pn > 1e-9: return (pruned * (on / (pn + 1e-9))).to(diff_tensor.dtype)
    else: return torch.zeros_like(diff_tensor)

def randomized_svd_torch(tensor, k):
    x = tensor.to(dtype=torch.float32, device=tensor.device)
    p = k + 16
    if min(x.shape) <= p:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False); r = min(k, S.numel())
        return U[:, :r], S[:r], Vh[:r, :]
    Q = torch.randn(x.shape[1], p, device=x.device, dtype=x.dtype)
    Z = x @ Q
    Qt, _ = torch.linalg.qr(Z)
    Tt = Qt.T @ x
    Ut, S, Vh = torch.linalg.svd(Tt, full_matrices=False)
    U = Qt @ Ut; r = min(k, S.numel())
    return U[:, :r], S[:r], Vh[:r, :]

def project_tensor(teacher_tensor, student_shape):
    dev = teacher_tensor.device; dt = teacher_tensor.dtype
    if teacher_tensor.dim() == 1:
        out = torch.zeros(student_shape, device=dev, dtype=dt)
        out[:min(teacher_tensor.numel(), student_shape[0])] = teacher_tensor[:min(teacher_tensor.numel(), student_shape[0])]
        return out
    if teacher_tensor.dim() == 2:
        try:
            to, ti = student_shape
            k = min(teacher_tensor.shape[0], teacher_tensor.shape[1], to, ti)
            U, S, Vh = randomized_svd_torch(teacher_tensor, k)
            M = U @ torch.diag(S) @ Vh
            final = torch.zeros(student_shape, device=dev, dtype=dt)
            co, ci = min(M.shape[0], to), min(M.shape[1], ti)
            final[:co, :ci] = M[:co, :ci]
            return final
        except (torch.linalg.LinAlgError, torch.OutOfMemoryError):
            try:
                cpu = teacher_tensor.detach().to("cpu", dtype=torch.float32)
                U, S, Vh = torch.linalg.svd(cpu, full_matrices=False)
                to, ti = student_shape; k = min(len(S), to, ti)
                M = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
                final = torch.zeros(student_shape, dtype=torch.float32)
                co, ci = min(M.shape[0], to), min(M.shape[1], ti)
                final[:co, :ci] = M[:co, :ci]
                return final.to(dev, dtype=dt)
            except Exception:
                fb = torch.zeros(student_shape, device=dev, dtype=dt)
                cr, cc = min(teacher_tensor.shape[0], student_shape[0]), min(teacher_tensor.shape[1], student_shape[1])
                fb[:cr, :cc] = teacher_tensor[:cr, :cc]
                return fb
    return torch.zeros(student_shape, device=dev, dtype=dt)

def align_with_generalized_procrustes(src, tgt):
    src_f = torch.nan_to_num(src.float(), nan=0.0, posinf=0.0, neginf=0.0)
    tgt_f = torch.nan_to_num(tgt.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if src_f.shape != tgt_f.shape or src_f.dim() != 2: return src
    try:
        if torch.linalg.vector_norm(src_f) < 1e-8: return src
        R, _, _, _ = torch.linalg.lstsq(src_f, tgt_f)
        if not torch.all(torch.isfinite(R)): return src
        return (src_f @ R).to(src.dtype)
    except Exception:
        return src

def get_rank_for_key(key, shape, args):
    max_r = min(shape)
    if args.rank_moe and ("block_sparse_moe" in key or ".experts." in key):
        return min(args.rank_moe, max_r)
    if args.rank_attn and any(h in key for h in ATTN_HINTS):
        return min(args.rank_attn, max_r)
    if args.rank_mlp and any(h in key for h in MLP_HINTS):
        return min(args.rank_mlp, max_r)
    return min(args.rank_default, max_r)

def extract_lora_from_diff(diff, rank):
    if diff.dim() != 2: return None, None, "NOT_2D"
    if not torch.all(torch.isfinite(diff)): return None, None, "NAN_TENSOR"
    if torch.linalg.vector_norm(diff.float()) < 1e-8: return None, None, "ZERO_NORM"
    try:
        U, S, Vh = randomized_svd_torch(diff, rank)
    except Exception:
        try:
            cpu = diff.detach().to("cpu", dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(cpu, full_matrices=False)
            U, S, Vh = U.to(diff.device), S.to(diff.device), Vh.to(diff.device)
        except Exception:
            return None, None, "SVD_FAIL"
    lora_A = Vh.contiguous(); lora_B = (U @ torch.diag(S)).contiguous()
    return lora_A.to(torch.bfloat16), lora_B.to(torch.bfloat16), "SUCCESS"

# =================================================================================
#                             LAYER DEPTH MAPPING (KD)
# =================================================================================

def teacher_idx_from_student_idx(student_idx, student_layers, teacher_layers, schedule="sigmoid", k=0.15):
    if student_layers <= 1 or teacher_layers <= 1: return 0, 1.0
    if schedule == "linear":
        s_norm = student_idx / (student_layers - 1); t_float = s_norm * (teacher_layers - 1)
    else:
        to_norm = lambda idx, total: 2 * (idx / (total - 1)) - 1
        from_norm = lambda x, total: (x + 1) * (total - 1) / 2
        s_norm = to_norm(student_idx, student_layers)
        t_norm = math.tanh(s_norm / k) / math.tanh(1 / k)
        t_float = from_norm(t_norm, teacher_layers)
    t_floor = int(math.floor(t_float)); t_ceil = min(t_floor + 1, teacher_layers - 1)
    interp_w = t_float - t_floor
    return t_floor, interp_w

# =================================================================================
#                                   MoE DISTILL
# =================================================================================

def kmeans_or_fallback(X_np, k, niter=25, use_gpu=True):
    n, d = X_np.shape
    if faiss is not None and use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        km = faiss.Kmeans(d, k, niter=niter, verbose=False, gpu=res); km.train(X_np)
        D, I = km.index.search(X_np, 1); return km.centroids, I.flatten()
    if faiss is not None and not use_gpu:
        km = faiss.Kmeans(d, k, niter=niter, verbose=False); km.train(X_np)
        D, I = km.index.search(X_np, 1); return km.centroids, I.flatten()
    # Fallback: uniform bins
    idxs = np.arange(n, dtype=np.int64); bins = np.array_split(idxs, k)
    labels = np.zeros(n, dtype=np.int64)
    for ci, b in enumerate(bins): labels[b] = ci
    centroids = np.stack([X_np[b].mean(axis=0) if len(b) else np.zeros(d, dtype=X_np.dtype) for b in bins], axis=0)
    return centroids, labels

def distill_moe_layer(student_layer_idx, cfg, teacher_folder, teacher_weight_map, student_folder, student_weight_map, device, teacher_prefix_token, student_prefix_token, args):
    t_floor, interp_w = teacher_idx_from_student_idx(student_layer_idx, cfg['student_layers'], cfg['teacher_layers'], args.map_schedule, args.sigmoid_k)
    t_ceil = min(t_floor + 1, cfg['teacher_layers'] - 1)
    expert_parts = ['gate_proj', 'up_proj', 'down_proj', 'w1', 'w2', 'w3']
    batch_size = 64
    fingerprints = []

    for i in tqdm(range(0, cfg['teacher_experts_per_layer'], batch_size), desc=f"GPU {device[-1]} Fingerprint MoE L{student_layer_idx}", leave=False, dynamic_ncols=True):
        expert_indices = range(i, min(i + batch_size, cfg['teacher_experts_per_layer']))
        keys_to_fetch = []
        for l_idx in [t_floor, t_ceil]:
            for e_idx in expert_indices:
                for part in expert_parts:
                    keys_to_fetch.append(f"{teacher_prefix_token}.{l_idx}.block_sparse_moe.experts.{e_idx}.{part}.weight")
        batch_tensors_cpu = get_scaled_fp8_tensors(keys_to_fetch, teacher_folder, teacher_weight_map, device="cpu")
        for expert_idx in expert_indices:
            fp_floor_parts, fp_ceil_parts = [], []
            for part in expert_parts:
                key_floor = f"{teacher_prefix_token}.{t_floor}.block_sparse_moe.experts.{expert_idx}.{part}.weight"
                key_ceil  = f"{teacher_prefix_token}.{t_ceil}.block_sparse_moe.experts.{expert_idx}.{part}.weight"
                if key_floor in batch_tensors_cpu and key_ceil in batch_tensors_cpu:
                    t_floor_gpu = batch_tensors_cpu[key_floor].to(device, non_blocking=True)
                    t_ceil_gpu  = batch_tensors_cpu[key_ceil].to(device, non_blocking=True)
                    fp_floor_parts.append(t_floor_gpu.flatten()); fp_ceil_parts.append(t_ceil_gpu.flatten())
            if fp_floor_parts and fp_ceil_parts:
                fp_floor = torch.cat(fp_floor_parts); fp_ceil = torch.cat(fp_ceil_parts)
                fingerprints.append(slerp(fp_floor, fp_ceil, interp_w).cpu())
        del batch_tensors_cpu; torch.cuda.empty_cache()

    if not fingerprints: return {}
    X = torch.stack(fingerprints).float().cpu().numpy()
    use_gpu = (faiss is not None) and torch.cuda.is_available() and (not args.fallback_uniform_moe)
    centroids, labels = kmeans_or_fallback(X, cfg['student_experts_per_layer'], niter=args.kmeans_iter, use_gpu=use_gpu)

    expert_map = defaultdict(list)
    for teacher_idx, cluster_id in enumerate(labels): expert_map[int(cluster_id)].append(int(teacher_idx))

    parts = ['gate_proj', 'up_proj', 'down_proj', 'w1', 'w2', 'w3']
    student_shapes = {}
    for part in parts:
        key0 = f"{student_prefix_token}.{student_layer_idx}.block_sparse_moe.experts.0.{part}.weight"
        maybe = get_tensors_from_shards([key0], student_folder, student_weight_map, device="meta").get(key0)
        if maybe is not None: student_shapes[part] = maybe.shape

    synthetic = {}
    for s_e in tqdm(range(cfg['student_experts_per_layer']), desc=f"GPU {device[-1]} Blend MoE L{student_layer_idx}", leave=False, dynamic_ncols=True):
        assigned = expert_map.get(s_e, [])
        if len(assigned) == 0: continue
        assigned_fps = torch.from_numpy(X[assigned])
        centroid = torch.from_numpy(centroids[s_e]).unsqueeze(0)
        sim = -torch.sum((assigned_fps - centroid)**2, dim=1)
        num_to_blend = min(len(assigned), args.max_teachers_to_blend)
        topk = torch.topk(sim, k=num_to_blend)
        chosen = [assigned[i] for i in topk.indices.tolist()]
        blend_weights = F.softmax(topk.values, dim=0)

        keys_to_fetch = []
        for teacher_idx in chosen:
            for l_idx in [t_floor, t_ceil]:
                for part in parts:
                    keys_to_fetch.append(f"{teacher_prefix_token}.{l_idx}.block_sparse_moe.experts.{teacher_idx}.{part}.weight")
        blend_tensors_cpu = get_scaled_fp8_tensors(keys_to_fetch, teacher_folder, teacher_weight_map, device="cpu")

        for part, shape in student_shapes.items():
            interp_tensors = []
            for teacher_idx in chosen:
                key_floor = f"{teacher_prefix_token}.{t_floor}.block_sparse_moe.experts.{teacher_idx}.{part}.weight"
                key_ceil  = f"{teacher_prefix_token}.{t_ceil}.block_sparse_moe.experts.{teacher_idx}.{part}.weight"
                if key_floor in blend_tensors_cpu and key_ceil in blend_tensors_cpu:
                    t_floor = blend_tensors_cpu[key_floor].to(device)
                    t_ceil  = blend_tensors_cpu[key_ceil].to(device)
                    interp = slerp(t_floor, t_ceil, interp_w)
                    projected = project_tensor(interp, shape)
                    interp_tensors.append(projected)
            if not interp_tensors: continue
            try:
                stacked = torch.stack(interp_tensors).to(dtype=torch.float32)
                num_experts, rows, cols = stacked.shape
                flat = stacked.view(num_experts, -1)
                k_svd = min(128, num_experts)
                _, _, Vh = randomized_svd_torch(flat, k=k_svd)
                mean_flat = torch.mean(flat, dim=0)
                proj = mean_flat @ Vh.T; recon_flat = proj @ Vh
                synthesized = recon_flat.view(rows, cols)
            except Exception:
                weights_reshaped = blend_weights.to(stacked.device).view(-1, 1, 1)
                synthesized = torch.sum(stacked * weights_reshaped, dim=0)
            synthetic[f"{student_prefix_token}.{student_layer_idx}.block_sparse_moe.experts.{s_e}.{part}.weight"] = synthesized.cpu()

        del blend_tensors_cpu; torch.cuda.empty_cache()

    return synthetic

# ---------- NEW: Teacher MoE (mlp.experts) -> Student Dense MLP synthesis ----------

def count_teacher_mlp_experts_for_layer(teacher_weight_map, teacher_prefix_token, layer_idx):
    # Accept both 'mlp.experts' and 'block_sparse_moe.experts'
    pat_a = re.compile(re.escape(f"{teacher_prefix_token}.{layer_idx}.mlp.experts.") + r"(\d+)\.")
    pat_b = re.compile(re.escape(f"{teacher_prefix_token}.{layer_idx}.block_sparse_moe.experts.") + r"(\d+)\.")
    seen = set()
    for k in teacher_weight_map.keys():
        m = pat_a.search(k) or pat_b.search(k)
        if m: seen.add(int(m.group(1)))
    return (max(seen) + 1) if seen else 0

def synthesize_dense_mlp_from_teacher_moe(student_layer_idx, teacher_layers, student_layers,
                                          teacher_folder, teacher_weight_map, device,
                                          teacher_prefix_token, args):
    """
    Build a dense MLP (gate/up/down) for the student by aggregating teacher experts.
    Returns dict { 'gate_proj': tensor, 'up_proj': tensor, 'down_proj': tensor } on CPU.
    """
    t_floor, interp_w = teacher_idx_from_student_idx(student_layer_idx, student_layers, teacher_layers, args.map_schedule, args.sigmoid_k)
    t_ceil = min(t_floor + 1, teacher_layers - 1)

    # How many experts?
    t_experts = count_teacher_mlp_experts_for_layer(teacher_weight_map, teacher_prefix_token, t_floor)
    if t_experts == 0:
        return {}

    # Read router (if present): mlp.gate.weight (num_experts, hidden_size)
    gate_keys = [f"{teacher_prefix_token}.{t_floor}.mlp.gate.weight",
                 f"{teacher_prefix_token}.{t_ceil}.mlp.gate.weight"]
    router = get_tensors_from_shards(gate_keys, teacher_folder, teacher_weight_map, device="cpu")
    def router_scores(t):
        # L2 norm per expert row -> softmax -> importance
        if t is None: return None
        r = torch.linalg.vector_norm(t.float(), dim=1)  # [E]
        return F.softmax(r, dim=0)

    w_floor = router_scores(router.get(gate_keys[0], None))
    w_ceil  = router_scores(router.get(gate_keys[1], None))

    # Combine floor/ceil expert weights
    if w_floor is None and w_ceil is None:
        # Uniform if no router
        w_comb = torch.full((t_experts,), 1.0 / t_experts)
    else:
        if w_floor is None: w_floor = torch.full((t_experts,), 1.0 / t_experts)
        if w_ceil  is None: w_ceil  = torch.full((t_experts,), 1.0 / t_experts)
        w_comb = (1.0 - interp_w) * w_floor + interp_w * w_ceil  # [E]
        w_comb = w_comb / (w_comb.sum() + 1e-12)

    # Optionally take top-K for efficiency
    topk = min(int(args.max_teachers_to_blend), t_experts)
    vals, idxs = torch.topk(w_comb, k=topk)
    vals = (vals / (vals.sum() + 1e-12)).tolist()
    selected = idxs.tolist()

    parts = ['gate_proj', 'up_proj', 'down_proj', 'w1', 'w2', 'w3']  # support both styles
    blend = { 'gate_proj': None, 'up_proj': None, 'down_proj': None, 'w1': None, 'w2': None, 'w3': None }

    # Gather needed keys
    keys = []
    for e in selected:
        for l in [t_floor, t_ceil]:
            for p in parts:
                # try qwen-style first
                keys.append(f"{teacher_prefix_token}.{l}.mlp.experts.{e}.{p}.weight")
                # and mixtral-style as backup
                keys.append(f"{teacher_prefix_token}.{l}.block_sparse_moe.experts.{e}.{p}.weight")
    bag = get_scaled_fp8_tensors(keys, teacher_folder, teacher_weight_map, device="cpu")

    # Aggregate per part
    for p in parts:
        comp = None
        for w, e in zip(vals, selected):
            # prefer qwen-style key; fallback to block_sparse
            kf_a = f"{teacher_prefix_token}.{t_floor}.mlp.experts.{e}.{p}.weight"
            kc_a = f"{teacher_prefix_token}.{t_ceil}.mlp.experts.{e}.{p}.weight"
            kf_b = f"{teacher_prefix_token}.{t_floor}.block_sparse_moe.experts.{e}.{p}.weight"
            kc_b = f"{teacher_prefix_token}.{t_ceil}.block_sparse_moe.experts.{e}.{p}.weight"
            tf = bag.get(kf_a, bag.get(kf_b, None))
            tc = bag.get(kc_a, bag.get(kc_b, None))
            if tf is None or tc is None: continue
            inter = slerp(tf, tc, interp_w)  # [out,in]
            comp = inter * w if comp is None else comp + inter * w
        if comp is not None:
            blend[p] = comp.cpu()

    # Map 'w1/w2/w3' back to (gate/up/down) if needed: prefer gate/up/down already filled
    # If only w1/2/3 exist, alias them as gate/up/down (heuristic for SwiGLU variants)
    if blend['gate_proj'] is None and blend['w1'] is not None: blend['gate_proj'] = blend['w1']
    if blend['up_proj']   is None and blend['w3'] is not None: blend['up_proj']   = blend['w3']
    if blend['down_proj'] is None and blend['w2'] is not None: blend['down_proj'] = blend['w2']

    # Keep only requested three parts
    out = {k:v for k,v in blend.items() if k in ('gate_proj','up_proj','down_proj') and v is not None}
    return out

# =================================================================================
#                                  WORKER LOGIC
# =================================================================================

def distill_worker(rank, world_size, args, student_keys, teacher_weight_map, student_weight_map,
                   teacher_layers_map, student_layers_map, teacher_prefix_guess, student_prefix_guess):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if device != "cpu": torch.cuda.set_device(device)
    print(f"--- Worker {rank} on {device} ---")

    worker_lora = {}; debug_stats = defaultdict(int)
    keys_by_shard = defaultdict(list)
    for k in student_keys:
        if k in student_weight_map: keys_by_shard[student_weight_map[k]].append(k)

    # cache synthesized MLP per layer to avoid triple work
    mlp_cache = {}  # (layer_idx) -> {'gate_proj': T, 'up_proj': T, 'down_proj': T}

    with tqdm(total=len(student_keys), desc=f"GPU {rank} Init", position=rank, dynamic_ncols=True, unit="keys") as pbar:
        for shard_file, keys_in_shard in sorted(keys_by_shard.items()):
            pbar.set_description(f"GPU {rank} | Shard: {os.path.basename(shard_file)}")
            for i in range(0, len(keys_in_shard), args.micro_bs):
                micro_keys = keys_in_shard[i:i+args.micro_bs]
                student_tensors_cpu = get_tensors_from_shards(micro_keys, args.student, student_weight_map, device="cpu")

                teacher_lookup = defaultdict(list)
                all_teacher_keys_batch = set()

                for sk in micro_keys:
                    if not sk.endswith(".weight"): continue
                    if any(sk.endswith(suf) for suf in SKIP_SUFFIXES): continue
                    if any(tok in sk.lower() for tok in NORM_TOKENS): continue
                    if (not args.include_embed_lm_head) and any(tok in sk for tok in EMBED_TOKENS): continue
                    if args.include and not re.search(args.include, sk): continue
                    if args.exclude and re.search(args.exclude, sk): continue

                    seg = split_key(sk)
                    if seg is None:
                        tk = build_same_suffix_key(sk, student_prefix_guess, teacher_prefix_guess)
                        if tk in teacher_weight_map:
                            teacher_lookup[sk].append(tk); all_teacher_keys_batch.add(tk)
                        continue

                    s_prefix, token_name, s_idx, rest = seg
                    token_fmt = token_name + ".{idx}"
                    s_layers = len(student_layers_map.get(token_name, []))
                    t_layers = len(teacher_layers_map.get(token_name, []))
                    if s_layers == 0 or t_layers == 0:
                        token_name, s_layers, t_layers = get_layer_count_for_key(sk, teacher_layers_map, student_layers_map)
                        token_fmt = token_name + ".{idx}"

                    t_floor, interp_w = teacher_idx_from_student_idx(s_idx, s_layers, t_layers, args.map_schedule, args.sigmoid_k)
                    t_ceil = min(t_floor + 1, t_layers - 1)

                    key_floor = f"{s_prefix}{token_fmt.format(idx=t_floor)}.{rest}"
                    key_ceil  = f"{s_prefix}{token_fmt.format(idx=t_ceil)}.{rest}"
                    if key_floor.startswith(student_prefix_guess):
                        key_floor = teacher_prefix_guess + key_floor[len(student_prefix_guess):]
                    if key_ceil.startswith(student_prefix_guess):
                        key_ceil = teacher_prefix_guess + key_ceil[len(student_prefix_guess):]

                    # direct teacher matches (dense<->dense or attn)
                    if key_floor in teacher_weight_map:
                        teacher_lookup[sk].append(key_floor); all_teacher_keys_batch.add(key_floor)
                    if key_ceil in teacher_weight_map and key_ceil != key_floor:
                        teacher_lookup[sk].append(key_ceil); all_teacher_keys_batch.add(key_ceil)

                teacher_tensors_cpu = get_scaled_fp8_tensors(list(all_teacher_keys_batch), args.teacher, teacher_weight_map, device="cpu")

                for sk in micro_keys:
                    pbar.update(1)
                    st = student_tensors_cpu.get(sk, None)
                    if st is None or not (sk.endswith(".weight")): continue

                    # --------- MoE->Dense MLP special-case (teacher has experts, student is dense) ---------
                    # Trigger when: key is mlp.{gate/up/down}_proj on student, but we didn't find a direct teacher match.
                    if (".mlp." in sk) and (".experts." not in sk) and (len(teacher_lookup.get(sk, [])) == 0):
                        seg = split_key(sk)
                        if seg is not None:
                            s_prefix, token_name, s_idx, rest = seg
                            s_layers = len(student_layers_map.get(token_name, []))
                            t_layers = len(teacher_layers_map.get(token_name, []))
                            if s_layers > 0 and t_layers > 0:
                                t_prefix_token = teacher_prefix_guess + token_name  # e.g., 'model.layers'
                                # synthesize per-layer MLP once
                                if s_idx not in mlp_cache:
                                    mlp_cache[s_idx] = synthesize_dense_mlp_from_teacher_moe(
                                        s_idx, t_layers, s_layers, args.teacher, teacher_weight_map, device,
                                        t_prefix_token, args
                                    )
                                part = None
                                if ".mlp.gate_proj.weight" in sk: part = "gate_proj"
                                elif ".mlp.up_proj.weight" in sk: part = "up_proj"
                                elif ".mlp.down_proj.weight" in sk: part = "down_proj"
                                else:
                                    # also support w1/w2/w3-named students (rare)
                                    if ".mlp.w1.weight" in sk: part = "gate_proj"
                                    if ".mlp.w3.weight" in sk: part = "up_proj"
                                    if ".mlp.w2.weight" in sk: part = "down_proj"
                                synth = mlp_cache.get(s_idx, {}).get(part, None)
                                if synth is not None:
                                    student_tensor = st.to(device)
                                    projected = project_tensor(synth.to(device), student_tensor.shape)
                                    aligned = align_with_generalized_procrustes(projected, student_tensor)
                                    diff = apply_dare_ties(aligned.float() - student_tensor.float())
                                    r = get_rank_for_key(sk, student_tensor.shape, args)
                                    la, lb, reason = extract_lora_from_diff(diff, r)
                                    debug_stats[reason] += 1
                                    if reason == "SUCCESS":
                                        worker_lora[f"{to_lora_key(sk, student_prefix_guess)}.lora_A.weight"] = la.cpu()
                                        worker_lora[f"{to_lora_key(sk, student_prefix_guess)}.lora_B.weight"] = lb.cpu()
                                    continue  # handled
                        # if failed to synthesize, fall through to default logic

                    # --------- Existing MoE student route ---------
                    if "block_sparse_moe" in sk and ".experts." in sk:
                        seg = split_key(sk)
                        if seg is None: continue
                        s_prefix, token_name, s_idx, _ = seg
                        s_layers = len(student_layers_map.get(token_name, []))
                        t_layers = len(teacher_layers_map.get(token_name, []))

                        def count_experts(weight_map, folder_prefix, layer_idx):
                            pat = re.compile(re.escape(folder_prefix + f".{layer_idx}.block_sparse_moe.experts.") + r"(\d+)\.")
                            seen = set()
                            for k in weight_map.keys():
                                m = pat.search(k)
                                if m: seen.add(int(m.group(1)))
                            return (max(seen) + 1) if seen else 0

                        t_prefix_token = teacher_prefix_guess + token_name
                        s_prefix_token = student_prefix_guess + token_name
                        t_experts = count_experts(teacher_weight_map, t_prefix_token, 0)
                        s_experts = count_experts(student_weight_map, s_prefix_token, 0)
                        cfg = dict(teacher_layers=t_layers, student_layers=s_layers,
                                   teacher_experts_per_layer=t_experts, student_experts_per_layer=s_experts)
                        moe_weights = distill_moe_layer(s_idx, cfg, args.teacher, teacher_weight_map,
                                                        args.student, student_weight_map, device,
                                                        t_prefix_token, s_prefix_token, args)
                        if moe_weights:
                            for key, synth_cpu in moe_weights.items():
                                student_tensor = get_tensors_from_shards([key], args.student, student_weight_map, device=device).get(key, None)
                                if student_tensor is None: continue
                                aligned = align_with_generalized_procrustes(synth_cpu.to(device), student_tensor)
                                diff = apply_dare_ties(aligned.float() - student_tensor.float())
                                r = get_rank_for_key(key, student_tensor.shape, args)
                                la, lb, reason = extract_lora_from_diff(diff, r)
                                debug_stats[reason] += 1
                                if reason == "SUCCESS":
                                    worker_lora[f"{to_lora_key(key, student_prefix_guess)}.lora_A.weight"] = la.cpu()
                                    worker_lora[f"{to_lora_key(key, student_prefix_guess)}.lora_B.weight"] = lb.cpu()
                        continue  # finished MoE layer

                    # --------- Dense path (default) ---------
                    if any(tok in sk.lower() for tok in NORM_TOKENS): continue
                    if (not args.include_embed_lm_head) and any(tok in sk for tok in EMBED_TOKENS): continue
                    student_tensor = st.to(device)
                    tk = teacher_lookup.get(sk, [])
                    if not tk:
                        debug_stats["NO_TEACHER_MATCH"] += 1
                        continue

                    synth = None
                    if len(tk) == 1:
                        tt = teacher_tensors_cpu.get(tk[0], None)
                        if tt is not None: synth = project_tensor(tt.to(device), student_tensor.shape)
                    elif len(tk) >= 2:
                        t0 = teacher_tensors_cpu.get(tk[0], None); t1 = teacher_tensors_cpu.get(tk[1], None)
                        if t0 is not None and t1 is not None:
                            seg = split_key(sk)
                            if seg is not None:
                                _, token_name, s_idx, _ = seg
                                s_layers = len(student_layers_map.get(token_name, []))
                                t_layers = len(teacher_layers_map.get(token_name, []))
                                _, w = teacher_idx_from_student_idx(s_idx, s_layers, t_layers, args.map_schedule, args.sigmoid_k)
                                interp_w = w
                            else:
                                interp_w = 0.5
                            synth = project_tensor(slerp(t0.to(device), t1.to(device), interp_w), student_tensor.shape)

                    if synth is not None and is_linear_2d_shape(student_tensor):
                        aligned = align_with_generalized_procrustes(synth, student_tensor)
                        diff = apply_dare_ties(aligned.float() - student_tensor.float())
                        r = get_rank_for_key(sk, student_tensor.shape, args)
                        la, lb, reason = extract_lora_from_diff(diff, r)
                        debug_stats[reason] += 1
                        if reason == "SUCCESS":
                            worker_lora[f"{to_lora_key(sk, student_prefix_guess)}.lora_A.weight"] = la.cpu()
                            worker_lora[f"{to_lora_key(sk, student_prefix_guess)}.lora_B.weight"] = lb.cpu()
                    else:
                        debug_stats["NO_SYNTH"] += 1

                del student_tensors_cpu, teacher_tensors_cpu
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    tmp_path = f"{args.out_lora}.worker{rank}.safetensors"
    if worker_lora: save_file(worker_lora, tmp_path)

    print(f"\n--- Worker {rank} Debugging Report ---")
    total_processed = sum(debug_stats.values())
    print(f"Total Events: {total_processed}")
    for reason, count in debug_stats.items():
        print(f"  - {reason}: {count}")
    print("-------------------------------------\n")

# =================================================================================
#                                        MAIN
# =================================================================================

def main():
    args = build_argparser().parse_args()
    seed_all(args.seed)
    if args.lora_alpha is None: args.lora_alpha = args.rank_default
    if args.out_adapter is None:
        base, _ = os.path.splitext(args.out_lora); args.out_adapter = base + ".adapter_config.json"

    student_weight_map = read_index_map(args.student)
    teacher_weight_map = read_index_map(args.teacher)

    student_keys = sorted(student_weight_map.keys())
    student_layers_map = scan_layers(student_keys)
    teacher_layers_map = scan_layers(teacher_weight_map.keys())

    student_prefix_guess = find_prefix(student_keys, token_example="layers")
    teacher_prefix_guess = find_prefix(list(teacher_weight_map.keys()), token_example="layers")

    print("--- Architecture Discovery ---")
    print(f" Student layer tokens: {{ {', '.join([f'{k}:{len(v)}' for k,v in student_layers_map.items()])} }}")
    print(f" Teacher layer tokens: {{ {', '.join([f'{k}:{len(v)}' for k,v in teacher_layers_map.items()])} }}")
    print(f" Student prefix guess: '{student_prefix_guess}' | Teacher prefix guess: '{teacher_prefix_guess}'")
    print("--------------------------------------------------------------------------------")

    mp.set_start_method("spawn", force=True)
    world_size = args.num_gpus or (torch.cuda.device_count() or 1); world_size = max(1, world_size)
    print(f"--- Spawning {world_size} worker(s) ---")

    keys_per_worker = len(student_keys) // world_size
    procs = []
    for rank in range(world_size):
        start = rank * keys_per_worker
        end = None if rank == world_size - 1 else (rank + 1) * keys_per_worker
        subset = student_keys[start:end]
        p = mp.Process(target=distill_worker, args=(rank, world_size, args, subset, teacher_weight_map, student_weight_map,
                                                    teacher_layers_map, student_layers_map,
                                                    teacher_prefix_guess, student_prefix_guess))
        p.start(); procs.append(p)
    for p in procs: p.join()

    print("\n--- Consolidating LoRA weights ---")
    final_lora_weights = {}
    for rank in range(world_size):
        tmp = f"{args.out_lora}.worker{rank}.safetensors"
        if os.path.exists(tmp):
            final_lora_weights.update(load_file(tmp))
            try: os.remove(tmp)
            except Exception: pass

    if not final_lora_weights:
        print("❌ CRITICAL: No LoRA weights were generated."); return

    os.makedirs(os.path.dirname(args.out_lora) or ".", exist_ok=True)
    save_file(final_lora_weights, args.out_lora)
    print(f"✅ Saved consolidated LoRA weights to: {args.out_lora}")

    target_modules = sorted(list(set(
        re.sub(r'\.lora_[AB]\.weight$', '', key) for key in final_lora_weights.keys()
    )))
    if not target_modules:
        print("❌ No target modules found; cannot emit adapter_config.json."); return

    adapter_config = {
        "base_model_name_or_path": args.student,
        "peft_type": "LORA",
        "r": args.rank_default,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "bias": "none"
    }
    with open(args.out_adapter, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved adapter config to: {args.out_adapter}")
    print("\n🎉 Universal Distillation Complete!")

if __name__ == "__main__":
    main()
