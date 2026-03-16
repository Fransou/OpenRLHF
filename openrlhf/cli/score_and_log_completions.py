"""
A script for scoring generated completions and logging metrics to wandb.

This script processes scored generations and logs aggregated metrics to wandb.
"""

import argparse
import gc
import json
import re
import subprocess as sp
from pathlib import Path

import pandas as pd
import torch
import wandb

def process_scored_generations(scored_path: Path) -> pd.DataFrame:
    """
    Process scored generations from a jsonl file.

    Args:
        scored_path: Path to the scored generations jsonl file

    Returns:
        DataFrame with processed scored generations
    """
    # Open the jsonl file and read the scored generations
    step = re.search(r"global_step(\d+)_hf", scored_path.stem)
    if step is not None:
        step = int(step.group(1))
    else:
        step = 0
    dataframe = []

    with scored_path.open() as f:
        for line in f:
            item = json.loads(line)
            found_key = None
            for key in ["generation_verifier_metadata", "mol_prop_verifier_metadata", "reaction_verifier_metadata"]:
                if item["reward_meta"].get(key) is not None:
                    found_key = key
                    break
            if found_key is None:
                print("Warning: No metadata key found in reward_meta for item, skipping.")
                continue
            dataframe.append(item["reward_meta"][found_key])
            dataframe[-1]["reward"] = item["reward"]
            dataframe[-1]["prompt_id"] = item["metadata"].get("prompt_id", None)

            completion = item["output"]
            dataframe[-1]["n_tokens"] = (
                len(completion) // 4
            )  # Approximate token count (assuming 4 characters per token on average)

    # Convert the list of scored generations to a pandas DataFrame
    df = pd.DataFrame(dataframe)
    df["path"] = str(scored_path)
    df["global_step"] = step
    wandb_log_from_df(df, step)
    return df


def wandb_log_from_df(df: pd.DataFrame, step: int):
    """
    Log metrics from scored generations to wandb.

    Args:
        df: DataFrame with scored generations
        step: Global training step for logging
    """
    # Log mean-reward for the scored generations
    mean_reward = df.groupby("prompt_id")["reward"].mean().mean()
    mean_n_tokens = df.groupby("prompt_id")["n_tokens"].mean().mean()
    average_validity = df["smiles_extraction_failure"].apply(lambda x: int(x == "")).mean()

    def get_uniqueness(group: pd.DataFrame) -> float:
        smis = group[group["smiles_extraction_failure"] == ""]["all_smi"].apply(lambda x: x[-1])
        return smis.nunique() / len(smis) if len(smis) > 0 else 0.0

    average_uniqueness = df.groupby("prompt_id").apply(get_uniqueness).mean()

    wandb.log(
        {
            "mean_reward": mean_reward,
            "mean_n_tokens": mean_n_tokens,
            "average_validity": average_validity,
            "average_uniqueness": average_uniqueness,
        },
        step=step,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", "-c", type=str, required=True, help="Path to the directory containing model checkpoints"
    )
    parser.add_argument(
        "--subset",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of checkpoint steps to score. If provided, only checkpoints with these steps (e.g. global_step100_hf -> 100) will be scored.",
    )
    args = parser.parse_args()

    path = Path(args.ckpt_path)
    wandb.init(project="openrlhf-eval", name=f"eval_{path.parent.stem}")
    dfs = []

    # Sort checkpoints by global_step number for consistent wandb logging
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
        print(f"\n{'='*80}")
        print(f"Scoring completions for checkpoint: {ckpt}")
        print(f"{'='*80}")

        out_path = str(ckpt.parent / f"{ckpt.stem}_eval_results")

        print(f"Scoring results for checkpoint: {ckpt}")
        cmd = [
            "python",
            "-m",
            "mol_gen_docking.score_completions",
            "--input_file",
            out_path + ".jsonl",
            "--mol-generation",
        ]
        print(f"Scoring generations with command: {' '.join(cmd)}")
        sp.run(cmd)

        scored_path = Path(out_path + "_scored.jsonl")
        if scored_path.exists():
            print(f"Processing scored generations for checkpoint: {ckpt}")
            df = process_scored_generations(scored_path)
            dfs.append(df)
        else:
            print(f"Scored generations file not found for checkpoint: {ckpt}, expected at: {scored_path}")

    if dfs:
        print(f"\n{'='*80}")
        print("Aggregating results and logging to wandb...")
        print(f"{'='*80}")
        all_results_df = pd.concat(dfs, ignore_index=True)
        all_results_df.to_csv(path / "all_scored_generations.csv", index=False)
        print(f"Saved all scored generations to: {path / 'all_scored_generations.csv'}")
        # Log to wandb as a Table
        wandb.log({"scored_generations": wandb.Table(dataframe=all_results_df)})
        print("Logged scored generations table to wandb.")
    else:
        print("No scored generations found to aggregate.")

