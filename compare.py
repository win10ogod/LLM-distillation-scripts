import gc
import torch
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, PreTrainedModel, BitsAndBytesConfig


def extract_layer_info(
    module_name: str, layer_parent_modules: set[str]
) -> tuple[int, str] | tuple[None, None]:
    parts = module_name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] in layer_parent_modules and parts[i + 1].isdigit():
            layer_num = int(parts[i + 1])
            submodule_path = ".".join(parts[i + 2 :]) if i + 2 < len(parts) else ""
            return layer_num, submodule_path
    return None, None


def collect_model_params(
    model: PreTrainedModel, layer_parent_modules: set[str]
) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    params_info = {}
    param_count = 0
    
    for name, module in model.named_modules():
        layer_num, submodule_path = extract_layer_info(name, layer_parent_modules)
        if layer_num is None:
            continue

        # collect parameters and move to CPU
        module_params = {}
        for param_name, param in module.named_parameters(recurse=False):
            module_params[param_name] = param.detach().cpu()
            param_count += 1

        if module_params:
            params_info.setdefault(layer_num, {}).setdefault(submodule_path, {}).update(
                module_params
            )
    
    print(f"  Collected {param_count} parameters from {len(params_info)} layers")
    return params_info


def compare_with_collected_params(
    params_a: dict[int, dict[str, dict[str, torch.Tensor]]],
    model_b: PreTrainedModel,
    layer_parent_modules: set,
) -> dict:
    differences = {}
    compared_count = 0
    identical_count = 0

    for name, module_b in model_b.named_modules():
        layer_num, submodule_path = extract_layer_info(name, layer_parent_modules)
        if layer_num is None:
            continue

        # get parameters from model_a
        model_a_params = params_a.get(layer_num, {}).get(submodule_path, {})

        for param_name, param_b in module_b.named_parameters(recurse=False):
            param_a = model_a_params.get(param_name)

            if param_a is None:
                print(f"Warning: {name}.{param_name} not found in model_a")
                continue

            # compare on CPU
            param_b_cpu = param_b.detach().cpu()

            if param_a.shape != param_b_cpu.shape:
                print(
                    f"Shape mismatch: {name}.{param_name} {param_a.shape} vs {param_b_cpu.shape}"
                )
                continue

            # calculate difference
            with torch.no_grad():
                diff = torch.abs(param_a.float() - param_b_cpu.float())
                avg_diff = torch.mean(diff).item()
                max_diff = torch.max(diff).item()
            
            compared_count += 1
            if avg_diff == 0.0 and max_diff == 0.0:
                identical_count += 1

            differences.setdefault(layer_num, {}).setdefault(submodule_path, {})[
                param_name
            ] = {"avg": avg_diff, "max": max_diff, "tensor": diff}
    
    print(f"  Compared {compared_count} parameters")
    print(f"  Identical: {identical_count}, Different: {compared_count - identical_count}")
    return differences


def print_differences(differences: dict, show_all: bool = False) -> None:
    total_params = 0
    different_params = 0
    
    for layer_num in sorted(differences.keys()):
        has_diff = False
        layer = differences[layer_num]

        for module_path in sorted(layer.keys()):
            module = layer[module_path]

            for param_name in sorted(module.keys()):
                diff = module[param_name]
                total_params += 1
                
                if diff["avg"] != 0.0 or diff["max"] != 0.0:
                    different_params += 1
                    if not has_diff:
                        print(f"\nLayer {layer_num}:")
                        has_diff = True
                    
                    print(f"  Module: {module_path or '<root>'}")
                    print(f"    > {param_name}:")
                    print(f"      Avg diff: {diff['avg']:.6e}")
                    print(f"      Max diff: {diff['max']:.6e}")
                elif show_all:
                    if not has_diff:
                        print(f"\nLayer {layer_num}:")
                        has_diff = True
                    print(f"  Module: {module_path or '<root>'}")
                    print(f"    > {param_name}: IDENTICAL")
    
    print(f"\n" + "="*50)
    print(f"Summary: {different_params}/{total_params} parameters differ")
    if different_params == 0:
        print("✓ Models are IDENTICAL")
    else:
        print(f"✗ Models have {different_params} different parameters")


def inspect_model_structure(model: PreTrainedModel, name: str) -> None:
    print(f"\n{name} structure sample:")
    count = 0
    for module_name, _ in model.named_modules():
        if count < 10:
            print(f"  {module_name}")
            count += 1
    print("  ...")


def main():
    parser = ArgumentParser()
    parser.add_argument("-a", type=str, required=True, help="Path to model A")
    parser.add_argument("-b", type=str, required=True, help="Path to model B")
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all parameters including identical ones"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Show model structure for debugging"
    )
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit precision using bitsandbytes",
    )
    quant.add_argument(
        "--load-in-8bit",
        action="store_true",
        default=False,
        help="Load model in 8-bit precision using bitsandbytes",
    )
    args = parser.parse_args()
    
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    print("Loading model A...")
    model_a = AutoModelForCausalLM.from_pretrained(
        args.a,
        device_map=args.device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    
    if args.inspect:
        inspect_model_structure(model_a, "Model A")
    
    layer_parent_modules = {"layers", "model", "model.layers", "h", "encoder", "decoder", "layer"}
    params_a = collect_model_params(model_a, layer_parent_modules)

    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading model B...")
    model_b = AutoModelForCausalLM.from_pretrained(
        args.b,
        device_map=args.device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    
    if args.inspect:
        inspect_model_structure(model_b, "Model B")

    print("Comparing models...")
    diff = compare_with_collected_params(params_a, model_b, layer_parent_modules)

    del model_b
    gc.collect()
    torch.cuda.empty_cache()

    print_differences(diff, show_all=args.show_all)


if __name__ == "__main__":
    main()