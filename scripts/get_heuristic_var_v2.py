import pandas as pd
from fire import Fire


def main(input_path, prefix="NUC"):
    assert prefix.upper() in ("NUC", "MEM")

    df = pd.read_parquet(input_path)

    cols = [col for col in df.columns if prefix.upper() in col.upper()]

    _ranges = df[cols].std(axis=0) * 2
    twosigmas = _ranges / 10
    variances = (twosigmas / 2) ** 2
    for value in variances.tolist():
        print(f"    - {value:.10f}")


if __name__ == "__main__":
    Fire(main)
