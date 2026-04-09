import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from areal.dataset import get_custom_dataset
from areal.learned_loki import enable_learned_loki_training, save_learned_loki_weight


def _load_model(model_path: str, dtype: torch.dtype, attn_implementation: str):
    common_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            **common_kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            **common_kwargs,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect calibration keys and build a PCA-initialized Learned-Loki checkpoint."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, default="rl", choices=["rl", "sft"])
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--sink-window-size", type=int, default=16)
    parser.add_argument("--recent-window-size", type=int, default=64)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--threshold-init", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=16,
        help="Call torch.cuda.empty_cache() every N samples. Set 0 to disable.",
    )
    return parser.parse_args()


@dataclass
class LayerPCAStats:
    sum_x: torch.Tensor
    sum_xxt: torch.Tensor
    count: int

    @classmethod
    def create(cls, head_dim: int):
        return cls(
            sum_x=torch.zeros(head_dim, dtype=torch.float64),
            sum_xxt=torch.zeros(head_dim, head_dim, dtype=torch.float64),
            count=0,
        )

    def update(self, vectors: torch.Tensor):
        vectors = vectors.reshape(-1, vectors.shape[-1]).to(dtype=torch.float64, device="cpu")
        self.sum_x += vectors.sum(dim=0)
        self.sum_xxt += vectors.T @ vectors
        self.count += int(vectors.shape[0])

    def principal_components(self, rank: int) -> torch.Tensor:
        if self.count <= 0:
            raise ValueError("No calibration vectors were collected for a layer.")
        if rank > self.sum_x.shape[0]:
            raise ValueError(
                f"Requested rank {rank} exceeds the attention head dimension {self.sum_x.shape[0]}."
            )
        mean = self.sum_x / self.count
        covariance = self.sum_xxt / self.count - torch.outer(mean, mean)
        _, eigvecs = torch.linalg.eigh(covariance)
        top_components = eigvecs[:, -rank:].T.contiguous()
        return top_components.to(dtype=torch.float32)


def _sample_to_input_ids(sample, tokenizer, max_length: int) -> torch.Tensor:
    if "input_ids" in sample:
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0)
    elif "messages" in sample:
        if tokenizer.chat_template is not None:
            input_ids = tokenizer.apply_chat_template(
                sample["messages"],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            text = "\n".join(message["content"] for message in sample["messages"])
            input_ids = tokenizer(text, return_tensors="pt").input_ids
    elif "prompt" in sample:
        input_ids = tokenizer(sample["prompt"], return_tensors="pt").input_ids
    elif "question" in sample:
        input_ids = tokenizer(sample["question"], return_tensors="pt").input_ids
    else:
        raise ValueError(f"Unsupported calibration sample format: {sample.keys()}")

    if max_length is not None and input_ids.shape[-1] > max_length:
        input_ids = input_ids[:, :max_length]
    return input_ids


def _extract_key_tensors(past_key_values) -> List[torch.Tensor]:
    if past_key_values is None:
        raise ValueError("Model did not return past_key_values during calibration.")

    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.key_cache)

    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    if isinstance(past_key_values, (list, tuple)):
        if len(past_key_values) == 0:
            return []
        if isinstance(past_key_values[0], (list, tuple)):
            return [layer_cache[0] for layer_cache in past_key_values]

    raise ValueError(
        f"Unsupported past_key_values type for calibration: {type(past_key_values)}"
    )


def _iter_calibration_samples(dataset, limit: int) -> Iterable:
    for sample_idx, sample in enumerate(dataset):
        if sample_idx >= limit:
            break
        yield sample


def _collect_pca_stats(model, tokenizer, dataset, args) -> List[LayerPCAStats]:
    stats = None
    processed_samples = 0

    with torch.no_grad():
        for sample in tqdm(_iter_calibration_samples(dataset, args.num_samples), total=args.num_samples):
            input_ids = _sample_to_input_ids(sample, tokenizer, args.max_length).to(
                args.device
            )
            if input_ids.numel() == 0:
                continue

            outputs = model(input_ids=input_ids, use_cache=True)
            key_tensors = _extract_key_tensors(outputs.past_key_values)
            if stats is None:
                stats = [
                    LayerPCAStats.create(layer_keys.shape[-1])
                    for layer_keys in key_tensors
                ]

            for layer_idx, layer_keys in enumerate(key_tensors):
                stats[layer_idx].update(layer_keys.detach())

            processed_samples += 1
            del outputs, key_tensors, input_ids
            if (
                args.device.startswith("cuda")
                and args.empty_cache_every > 0
                and processed_samples % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    if stats is None or processed_samples == 0:
        raise ValueError("Calibration did not process any usable samples.")
    return stats


def _learned_loki_state_dict(model):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "learned_loki" in name
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )
    model = _load_model(
        model_path=args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()

    dataset = get_custom_dataset(
        path=args.dataset,
        rank=0,
        world_size=1,
        type=args.dataset_type,
        split=args.dataset_split,
        max_length=args.max_length,
        tokenizer=tokenizer,
    )

    pca_stats = _collect_pca_stats(model, tokenizer, dataset, args)

    enable_learned_loki_training(
        model,
        low_rank_dim=args.rank,
        budget_ratio=args.budget_ratio,
        sink_window_size=args.sink_window_size,
        recent_window_size=args.recent_window_size,
        gate_temperature=args.gate_temperature,
        threshold_init=args.threshold_init,
    )

    for layer_idx, layer in enumerate(model.model.layers):
        basis = pca_stats[layer_idx].principal_components(args.rank)
        module = layer.self_attn.learned_loki
        with torch.no_grad():
            module.weight.copy_(basis.to(device=module.weight.device, dtype=module.weight.dtype))
            module.threshold.fill_(args.threshold_init)
            module.gate_temperature = args.gate_temperature

    save_learned_loki_weight(model, _learned_loki_state_dict(model), args.output_dir)

    with open(os.path.join(args.output_dir, "calibration_meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved PCA-initialized Learned-Loki checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
