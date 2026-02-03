#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import os
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def safe_artifact_download_win(artifact):
    '''
    Safe way to download artifacts in windows as file path with ":" not accepted in windows. this function just updates local path with replacing ":" with "_"
    and saving it under "my_safe_artifacts" foldar in current dir instead temp
    '''
    artifact_dir = os.path.join(os.getcwd(), "my_safe_artifacts")
    print(f"Downloaded to: {artifact_dir}")
    artifact_folder = artifact.name.replace(":", "_")
    local_path = os.path.join(artifact_dir, artifact_folder)
    artifact_local_dir = artifact.download(root=local_path)
    return artifact_local_dir

def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    artifact = wandb.use_artifact(args.input_artifact)
    artifact_dir = safe_artifact_download_win(artifact)

    csv_files = [f for f in os.listdir(artifact_dir) if f.lower().endswith('.csv')]
    file_path = artifact_dir
    if len(csv_files) == 1:
        file_path = os.path.join(artifact_dir, csv_files[0])
        df = pd.read_csv(file_path)
        print(f"Loaded: {csv_files[0]}  ({len(df)} rows)")
    else:
        print(f"Found {len(csv_files)} CSV files — expected exactly 1")
        if csv_files:
            print("Files:", csv_files)
        else:
            print("No CSV file found")

    idx = df['price'].between(args.min_price, args.max_price)

    # Updated data with min amd Max price filter
    df = df[idx].copy()

    # Convert last_review to datetime
    #df['last_review'] = pd.to_datetime(df['last_review'])
    clean_csv_path = os.path.join(artifact_dir, "clean_sample.csv")
    df.to_csv(clean_csv_path, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(clean_csv_path)
    run.log_artifact(artifact)

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
