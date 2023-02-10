import pandas as pd
from fire import Fire


def main(input_path, prefix="NUC", hi=0.975, lo=0.025):
    assert prefix.upper() in ("NUC", "MEM")

    df = pd.read_parquet(input_path)

    cols = [col for col in df.columns if prefix.upper() in col.upper()]

    # heuristic to get the variance for the log-likelihood:
    #  - get a range of interest, defined by a high quantile and a low quantile
    #  - we assume the likelihood should be a gaussian with two-sigma equal to 1/10 of the found range
    #    - i.e. we should be within range/10 ~95% of the time
    #  - compute the corresponding variance value (i.e. the square of (twosigma / 2))
    _ranges = df[cols].quantile(hi) - df[cols].quantile(lo)
    twosigmas = _ranges / 10
    variances = (twosigmas / 2) ** 2
    for value in variances.tolist():
        print(f"    - {value:.10f}")


if __name__ == "__main__":
    Fire(main)
