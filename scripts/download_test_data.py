import shutil
from pathlib import Path

import boto3
import fire
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

EXAMPLE_DATA_DIR = Path(__file__).parent.parent/"data/example_experiment_data"
EXAMPLE_DATA_FILENAME = "s3_paths.csv"


def parse_s3_path(fn):
    path = Path(fn)
    assert path.parts[0] == "s3:", f"Expected an s3 path, got {fn}"
    return path.parts[1], "/".join(path.parts[2:]), path.parts[-1]


def setup_paths():
    for subdir in ("s3_data", "segmentation", "labelfree"):
        (EXAMPLE_DATA_DIR / subdir).mkdir(exist_ok=True, parents=True)


def download_test_data(limit=-1):
    setup_paths()

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    df = pd.read_csv(EXAMPLE_DATA_DIR / EXAMPLE_DATA_FILENAME)
    if limit > 0:
        df = df.iloc[:limit]

    local_data = {col: [] for col in df.columns}
    for col in df.columns:
        for fn in df[col].unique():
            bucket, s3_path, local_file_name = parse_s3_path(fn)
            local_file_path = EXAMPLE_DATA_DIR / "s3_data" / local_file_name
            if not local_file_path.exists():
                s3.download_file(bucket, s3_path, str(local_file_path))
            local_data[col].append(local_file_path)

    local_data = pd.DataFrame(local_data)
    labelfree_data = local_data[["raw", "raw"]]
    labelfree_data.columns = ["brightfield", "signal"]

    for data, data_type in zip((local_data, labelfree_data), ("segmentation", "labelfree")):
        for csv_type in ("train", "test", "valid"):
            data.to_csv(EXAMPLE_DATA_DIR / data_type / f"{csv_type}.csv")


def delete_test_data():
    for subdir in ("segmentation", "labelfree", "s3_data"):
        shutil.rmtree(EXAMPLE_DATA_DIR / subdir)


if __name__ == "__main__":
    fire.Fire(download_test_data)
