#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import pathlib
from pathlib import Path
import wandb
import os
import pandas as pd

import shutil

def empty_dir(dir_path: str) -> None:
    p = Path(dir_path)
    if not p.exists():
        return
    for child in p.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def safe_artifact_download_win(artifact, safe_root: str = "safe_artifacts") -> Path:
    """
    Safely downloads a wandb artifact on Windows by replacing ':' in names
    and storing it in a dedicated folder under the current working directory.

    Args:
        artifact: wandb.Artifact object
        safe_root: Name of the safe base folder (default: 'safe_artifacts')

    Returns:
        Path: Absolute path to the downloaded artifact directory

    Raises:
        ValueError: If artifact is invalid or download fails
    """
    if not isinstance(artifact, wandb.Artifact):
        raise ValueError("Input must be a valid wandb.Artifact object")

    # Get current working directory as base
    cwd = Path.cwd()

    # Create safe root folder if it doesn't exist
    safe_base = cwd / safe_root
    safe_base.mkdir(parents=True, exist_ok=True)

    # Replace ':' with '_' in artifact name (Windows can't handle ':')
    safe_name = artifact.name.replace(":", "_") if artifact.name else "unnamed_artifact"

    # Final download location
    target_dir = safe_base / safe_name

    print(f"Downloading artifact '{artifact.name}' to: {target_dir}")

    try:
        # Download the artifact
        downloaded_path = artifact.download(root=target_dir)

        # Convert to absolute Path object
        abs_path = Path(downloaded_path).resolve()

        print(f"Successfully downloaded to: {abs_path}")
        return abs_path

    except Exception as e:
        raise RuntimeError(f"Failed to download artifact '{artifact.name}': {e}")

def read_one_csv_to_df(artifact_inp_dir):
    """Reads the single CSV file in the directory — fails gracefully if not exactly one."""
    try:
        csv_files = [f for f in os.listdir(artifact_inp_dir)
                     if f.lower().endswith('.csv') and os.path.isfile(os.path.join(artifact_inp_dir, f))]

        if len(csv_files) != 1:
            msg = f"No CSV file found" if not csv_files else f"Found {len(csv_files)} CSVs — expected 1"
            print(msg + (f": {csv_files}" if csv_files else ""))
            return None

        file_path = os.path.join(artifact_inp_dir, csv_files[0])
        df = pd.read_csv(file_path)
        print(f"Loaded: {csv_files[0]} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"Error reading CSV from {artifact_inp_dir}: {e}")
        return None


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    artifact = wandb.use_artifact(args.input_artifact)
    artifact_inp_dir = safe_artifact_download_win(artifact,"my_safe_artifacts")

    df = read_one_csv_to_df(artifact_inp_dir)

    idx = df['price'].between(args.min_price, args.max_price)

    # Updated data with min amd Max price filter
    df = df[idx].copy()

    # Convert last_review to datetime
    #df['last_review'] = pd.to_datetime(df['last_review'])
    my_safe_dir = pathlib.Path(artifact_inp_dir).parent
    out_dir = my_safe_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    #out_dir = os.path.join(my_safe_dir, "output")
    clean_csv_path = os.path.join(out_dir,args.output_artifact)
    df.to_csv(clean_csv_path, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(clean_csv_path)
    run.log_artifact(artifact)
    empty_dir(my_safe_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact to be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Clean Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Clean Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Cleaned Output artifact that will be exported",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=int,
        help="Minimum Price to consider for clean data",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=int,
        help="Maximum Price to consider for clean data",
        required=True
    )


    args = parser.parse_args()

    go(args)
