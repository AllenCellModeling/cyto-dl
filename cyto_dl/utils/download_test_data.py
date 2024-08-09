import shutil
from pathlib import Path

import boto3
import pandas as pd
import pyrootutils
from botocore import UNSIGNED
from botocore.client import Config

EXAMPLE_DATA_DIR = None
EXAMPLE_DATA_FILENAME = "s3_paths.csv"


DATA_PATHS = {
    "raw": "s3://allencell/aics/variance_project_dataset/fov_path/008f53c7_3500001156_100X_20170807_1-Scene-09-P9-E07.ome.tiff",
    "seg": "s3://allencell/aics/variance_project_dataset/fov_seg_path/a7c64690_3500001156_100X_20170807_1-Scene-09-P9-E07_CellNucSegCombined.ome.tiff",
    "struct": "SEC61B",
}


def parse_s3_path(fn):
    path = Path(fn)
    assert path.parts[0] == "s3:", f"Expected an s3 path, got {fn}"
    return path.parts[1], "/".join(path.parts[2:]), path.parts[-1]


def setup_paths():
    global EXAMPLE_DATA_DIR
    root = pyrootutils.setup_root(
        search_from=__file__,
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=False,  # do NOT change working directory to root (would cause problems in DDP mode)
        indicator=("pyproject.toml", "README.md"),
    )
    EXAMPLE_DATA_DIR = root / "data" / "example_experiment_data"
    for subdir in ("s3_data", "segmentation", "labelfree"):
        (EXAMPLE_DATA_DIR / subdir).mkdir(exist_ok=True, parents=True)


def download_test_data(limit=-1):
    setup_paths()

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    df = pd.DataFrame(DATA_PATHS, index=[0])

    if limit > 0:
        df = df.iloc[:limit]

    local_data = {col: [] for col in df.columns}
    local_data["struct"] = DATA_PATHS["struct"]
    for col in ("raw", "seg"):
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
