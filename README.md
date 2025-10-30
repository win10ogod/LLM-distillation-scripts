# LLM Distillation Scripts

This repository provides a set of **Python scripts** for performing **Large Language Model (LLM) distillation**, including **Universal SVD-LoRA Distillation** for both dense and Mixture-of-Experts (MoE) architectures.
It also includes a comparison utility for verifying parameter-level differences between models.

---

## 📂 Repository Structure

```
LLM-distillation-scripts/
├── universal_distill_v2.py     # Core Universal SVD-LoRA distillation script
├── compare.py                  # Model comparison and layer-level difference inspection
├── README.md                   # Project description (this file)
└── examples/                   # (Optional) Example output or config files
```

---

## ⚙️ Key Features

* **Universal SVD-LoRA Distillation (v1.1)**
  Supports distilling LoRA adapters between dense and MoE models, including hybrid mappings.
* **Automatic Architecture Mapping**
  Detects layer structure and performs linear or sigmoid-based layer interpolation.
* **Dense ↔ MoE Conversion**
  Supports distilling MoE teachers (e.g., `mlp.experts`) into dense students (e.g., `mlp.gate_proj`).
* **Adapter Consolidation**
  Produces LoRA weights in `.safetensors` format and generates PEFT-compatible `adapter_config.json`.
* **Model Comparison Utility**
  Compares two Hugging Face models and outputs per-layer parameter differences.

---

## 🚀 Example Usage

### Distillation Command

```bash
python universal_distill_v2.py \
  --teacher models/cwm \
  --student models/qwen3-4b \
  --out_lora ./my-distill-cwm/adapter_model.safetensors \
  --num_gpus 1 \
  --micro_bs 16 \
  --rank_default 1024 \
  --map_schedule sigmoid \
  --sigmoid_k 0.15 \
  --include_embed_lm_head \
  --include "attn|mlp" \
  --exclude "norm|bias|rotary|rope"
```

This command performs a **LoRA distillation** from the teacher model (`cwm`) to the student (`qwen3-4b`) with sigmoid layer mapping and rank 1024.

---

### Model Comparison Command

```bash
python compare.py \
  -a models/base-model \
  -b ./my-distill-cwm \
  --device cuda \
  --show-all
```

This compares two models layer by layer, showing average and maximum parameter differences.

---

## 📦 Dependencies

Install the required Python packages:

```bash
pip install torch safetensors tqdm faiss-gpu transformers bitsandbytes
```

Optional for multi-GPU runs:

```bash
pip install accelerate
```

---

## 🧠 Distillation Output

After running `universal_distill_v2.py`, you’ll find:

* `adapter_model.safetensors` — LoRA adapter weights
* `adapter_config.json` — Configuration file for integration with PEFT or Transformers

---

## 📜 License

This project is released under the **Apache 2.0 License**.

---

## 🧩 Notes

* Fully compatible with **Hugging Face PEFT** and **transformers**.
* Automatically handles FP8 / FP16 quantized weights.
* Designed for research on **efficient LLM compression and adaptation**.
