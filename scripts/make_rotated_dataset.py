from pathlib import Path

import numpy as np
import pandas as pd
import pyrootutils
import torch
import yaml
from fire import Fire

from cyto_dl.transforms.dataframe import split_dataframe

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from cyto_dl.utils.spharm import batched_rotate_spharm, get_band_indices

BASE_VARIANCE = (
    "/allen/aics/modeling/VariancePlayground/orgmanifests/variance_dataset_07082022.csv"
)


def rotate_cols(df, cols, angles):
    band_indices = get_band_indices(16, cols, include_zero=False)
    with torch.no_grad():
        df[cols] = batched_rotate_spharm(
            torch.tensor(df[cols].values).float(), band_indices, angles
        ).numpy()
    return df


def main(output_dir):
    df = pd.read_csv(BASE_VARIANCE)

    nuc_shcoeff_cols = [
        col
        for col in df.columns
        if "shcoeff" in col and "nuc" in col.lower() and not (df[col] == 0).all()
    ]
    clean_nuc_cols = [
        "NUC_" + col[len("NUC_shcoeffs_") : -len("_lcc")] for col in nuc_shcoeff_cols
    ]

    df = df.rename(columns=dict(zip(nuc_shcoeff_cols, clean_nuc_cols)))

    mem_shcoeff_cols = [
        col
        for col in df.columns
        if "shcoeff" in col and "mem" in col.lower() and not (df[col] == 0).all()
    ]
    clean_mem_cols = [
        "MEM_" + col[len("MEM_shcoeffs_") : -len("_lcc")] for col in mem_shcoeff_cols
    ]

    df = df.rename(columns=dict(zip(mem_shcoeff_cols, clean_mem_cols)))

    cols = ["CellId"] + clean_nuc_cols + clean_mem_cols

    df = split_dataframe(df[cols], train_frac=0.7, val_frac=0.15, return_splits=False)

    df.to_parquet(Path(output_dir) / "aligned_manifest.parquet")

    random_angles = torch.tensor(np.random.uniform(-np.pi, np.pi, size=len(df))).float()

    df = rotate_cols(df, clean_nuc_cols, random_angles)
    df = rotate_cols(df, clean_mem_cols, random_angles)

    df["angles"] = random_angles.numpy()

    df.to_parquet(Path(output_dir) / "rotated_manifest.parquet")

    with open(Path(output_dir) / "base_spharm.yaml", "w") as f:
        yaml.dump({"mem_cols": clean_mem_cols, "nuc_cols": clean_nuc_cols}, f)


if __name__ == "__main__":
    Fire(main)
