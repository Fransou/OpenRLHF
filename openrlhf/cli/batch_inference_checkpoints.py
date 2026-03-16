"""
A script for generating completions for all checkpoints in a directory.

This script runs batch inference for each checkpoint and produces completion files.
"""

import argparse
import gc
import json
import os
import re
import subprocess as sp
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def clear_cuda_memory():
    """Clear CUDA memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def merge_lora_to_base(ckpt_path: Path) -> Path:
    """
    Merge LoRA weights with the base model and save to SLURM_TMPDIR.
    The base model path is retrieved from the adapter's config.

    Args:
        ckpt_path: Path to the LoRA checkpoint directory

    Returns:
        Path to the merged model saved in SLURM_TMPDIR
    """
    tmpdir = os.environ.get("SLURM_TMPDIR", "/tmp")
    merged_path = Path(tmpdir) / f"{ckpt_path.name}_merged"

    if merged_path.exists():
        print(f"Merged model already exists at {merged_path}, skipping merge.")
        return merged_path

    # Load the adapter config to get the base model path
    adapter_config_path = ckpt_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {ckpt_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(f"base_model_name_or_path not found in adapter_config.json at {ckpt_path}")

    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Loading LoRA weights from {ckpt_path}...")
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))

    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path}...")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    # Free memory
    del base_model, model, merged_model
    print(f"Merged model saved to {merged_path}")
    return merged_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--ckpt_path", "-c", type=str, required=True, help="Path to the directory containing model checkpoints"
    )
    parser.add_argument("--dashboard_port", type=int, default=8265, help="Port for Ray dashboard")

    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Whether to merge LoRA weights with the base model before evaluation (if applicable). This is required if the checkpoints are LoRA adapters and the batch inference script expects a full model.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of checkpoint steps to evaluate. If provided, only checkpoints with these steps (e.g. global_step100_hf -> 100) will be evaluated.",
    )
    args = parser.parse_args()

    path = Path(args.ckpt_path)

    # Sort checkpoints by global_step number for consistent processing
    checkpoints = sorted(
        path.glob("*_hf"),
        key=lambda p: (
            int(re.search(r"global_step(\d+)_hf", p.name).group(1)) if re.search(r"global_step(\d+)_hf", p.name) else 0
        ),
    )

    if args.subset:
        subset_steps = set(args.subset)
        checkpoints = [
            ckpt
            for ckpt in checkpoints
            if (
                int(re.search(r"global_step(\d+)_hf", ckpt.name).group(1))
                if re.search(r"global_step(\d+)_hf", ckpt.name)
                else 0
            )
            in subset_steps
        ]
        print(f"Filtered checkpoints to steps: {subset_steps}. Found {len(checkpoints)} matching checkpoints.")

    for ckpt in checkpoints:
        print(f"\n{'=' * 80}")
        print(f"Generating completions for checkpoint: {ckpt}")
        print(f"{'=' * 80}")

        # Merge LoRA weights with base model and save to SLURM_TMPDIR
        if args.merge_lora:
            merged_ckpt = merge_lora_to_base(ckpt)
        else:
            merged_ckpt = ckpt

        # Free CUDA memory after merging
        clear_cuda_memory()

        out_path = str(ckpt.parent / f"{ckpt.stem}_eval_results")
        cmd = [
            "ray",
            "job",
            "submit",
            f"--address=http://127.0.0.1:{args.dashboard_port}",
            '--runtime-env-json={"setup_commands": ["wandb offline"]}',
            "--",
            "python3",
            "-m",
            "openrlhf.cli.batch_inference",
            "--config",
            args.config,
            "--pretrain",
            str(merged_ckpt),
            "--output_path",
            out_path,
        ]
        print(f"Running batch inference with command: {' '.join(cmd)}")
        sp.run(cmd)

        # Free CUDA memory after batch inference
        # sleep for 10 s
        time.sleep(10)

        clear_cuda_memory()

    print(f"\n{'=' * 80}")
    print("Batch inference complete for all checkpoints!")
    print(f"{'=' * 80}")
