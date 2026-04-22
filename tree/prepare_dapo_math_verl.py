#!/usr/bin/env python3
"""Prepare DAPO-Math parquet for the standard verl RL dataset.

The raw local DAPO file keeps the chat-template-ready messages in
`source_prompt`, while `prompt` is a plain string. verl's RL dataset reads
`prompt` by default, so this script writes a new parquet where `prompt` is the
message array from `source_prompt`.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local DAPO-Math parquet to verl message-prompt format.")
    parser.add_argument(
        "--input",
        default="/root/autodl-tmp/data/dapo-math-17k/data/dapo-math-17k.parquet",
        help="Raw DAPO-Math parquet whose source_prompt contains chat messages.",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/data/dapo-math-17k/dapo-math-17k-verl.parquet",
        help="Output parquet with prompt replaced by source_prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_path}")

    dataframe = pd.read_parquet(input_path)
    required_columns = {"prompt", "source_prompt", "reward_model", "extra_info"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing_columns)}")
    if dataframe["source_prompt"].isna().any():
        raise ValueError("Invalid DAPO parquet: source_prompt contains null rows")

    converted = dataframe.copy()
    converted["prompt"] = converted["source_prompt"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(output_path, index=False)

    check = pd.read_parquet(output_path, columns=["prompt"])
    first_prompt = check.iloc[0]["prompt"]
    if isinstance(first_prompt, str):
        raise ValueError("Conversion failed: output prompt is still a string")

    print(f"Wrote {len(converted)} rows to {output_path}")
    print(f"First prompt type: {type(first_prompt).__name__}")


if __name__ == "__main__":
    main()
