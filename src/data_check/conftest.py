from pathlib import Path

import pytest
import pandas as pd
import wandb
import os

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

def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv)
    artifact_data_dir = safe_artifact_download_win(data_path)
    if artifact_data_dir is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = read_one_csv_to_df(artifact_data_dir)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref)
    artifact_ref_in = safe_artifact_download_win(data_path)
    if artifact_ref_in is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = df = read_one_csv_to_df(artifact_ref_in)

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)

@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
