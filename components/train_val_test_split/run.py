#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import os
import sys
# Add project root to sys.path so absolute imports from root work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import logging
import pandas as pd
import wandb
import tempfile
from pathlib import Path
from sklearn.model_selection import train_test_split


from components.wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

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

def go(args):

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_path = run.use_artifact(args.input)
    artifact_local_path = safe_artifact_download_win(artifact_path)
    df = read_one_csv_to_df(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files
    for df, k in zip([trainval, test], ['trainval', 'test']):
        filename = f"{k}_data.csv"
        local_path = os.path.join("safe_artifacts", filename)

        logger.info(f"Saving {filename} locally before upload")

        # Write directly to local file
        df.to_csv(local_path, index=False)

        logger.info(f"Uploading {filename} dataset")

        log_artifact(
            filename,
            f"{k}_data",
            f"{k} split of dataset",
            local_path,
            run,
        )
        # with tempfile.NamedTemporaryFile("w") as fp:
        #
        #     df.to_csv(fp.name, index=False)
        #
        #     log_artifact(
        #         f"{k}_data.csv",
        #         f"{k}_data",
        #         f"{k} split of dataset",
        #         fp.name,
        #         run,
        #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)
