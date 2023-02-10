import numpy as np
import pandas as pd
from aics_im2im.utils.spharm_rotation import get_band_indices
from fire import Fire


def main(input_path, prefix="NUC"):
    assert prefix.upper() in ("NUC", "MEM")

    df = pd.read_parquet(input_path)

    cols = [col for col in df.columns if prefix.upper() in col.upper()]

    power_spectrum = np.zeros((len(df), 16))
    power_per_coeff = np.zeros_like(df[cols].values)

    sqrs = df[cols].values ** 2
    band_indices = get_band_indices(16, cols, prefix, include_zero=True)
    for ix, band in enumerate(band_indices):
        power_spectrum[:, ix] = sqrs[:, band].sum(axis=1)
        per_coeff = power_spectrum[:, ix] / len(band)
        for m in band:
            power_per_coeff[:, m] = per_coeff

    power_per_coeff = power_per_coeff.mean(axis=0)
    power_per_coeff = power_per_coeff / power_per_coeff.max()
    power_per_coeff[0] = 1

    for value in power_per_coeff:
        print(f"    - {value:.10f}")


if __name__ == "__main__":
    Fire(main)
