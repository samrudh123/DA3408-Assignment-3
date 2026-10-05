import argparse
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

FLOAT_DECIMALS = 6


def normalize_type(t):
    if pa.types.is_large_string(t) or pa.types.is_string(t):
        return "string"
    if pa.types.is_timestamp(t):
        return "timestamp"
    return str(t)


def normalize_batch(batch: pa.RecordBatch) -> pd.DataFrame:
    df = batch.to_pandas(ignore_metadata=True)
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            df[col] = s.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[us]")
        elif pd.api.types.is_datetime64_any_dtype(s):
            df[col] = s.astype("datetime64[us]")
        elif pd.api.types.is_float_dtype(s):
            df[col] = s.round(FLOAT_DECIMALS)
        elif pd.api.types.is_integer_dtype(s):
            df[col] = s.astype("int64")
        else:
            df[col] = s.astype(str)
    return df


def profile(path):
    dataset = ds.dataset(path, format="parquet")
    schema = dataset.schema
    rows = 0
    fingerprint = np.uint64(0)
    nulls = {name: 0 for name in schema.names}
    sums = {}
    for batch in dataset.to_batches(columns=schema.names):
        rows += batch.num_rows
        for name in schema.names:
            nulls[name] += batch.column(name).null_count
        df = normalize_batch(batch)
        for name in df.select_dtypes("number").columns:
            sums[name] = sums.get(name, 0.0) + float(df[name].sum())
        hashes = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)
        with np.errstate(over="ignore"):
            fingerprint = fingerprint + hashes.sum(dtype=np.uint64)
    return {"schema": schema, "rows": rows, "nulls": nulls, "sums": sums, "fingerprint": fingerprint}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spark", default="output/spark_clean")
    parser.add_argument("--ray", default="output/ray_clean")
    args = parser.parse_args()

    spark, ray = profile(args.spark), profile(args.ray)
    failures = []

    spark_cols = [(f.name, normalize_type(f.type)) for f in spark["schema"]]
    ray_cols = [(f.name, normalize_type(f.type)) for f in ray["schema"]]
    print(f"{'column':<24}{'spark type':<16}{'ray type':<16}")
    for (sn, st), (rn, rt) in zip(spark_cols, ray_cols):
        flag = "" if (sn, st) == (rn, rt) else "  <-- MISMATCH"
        print(f"{sn:<24}{st:<16}{rt:<16}{flag}")
    if spark_cols != ray_cols:
        failures.append("schema")

    print(f"\nrows: spark={spark['rows']:,} ray={ray['rows']:,}")
    if spark["rows"] != ray["rows"]:
        failures.append("row count")

    for name in spark["nulls"]:
        if spark["nulls"][name] != ray["nulls"].get(name):
            failures.append(f"null count {name}")

    print(f"\n{'column':<24}{'spark sum':>22}{'ray sum':>22}")
    for name, s_sum in spark["sums"].items():
        r_sum = ray["sums"].get(name, float("nan"))
        ok = np.isclose(s_sum, r_sum, rtol=1e-9)
        print(f"{name:<24}{s_sum:>22.4f}{r_sum:>22.4f}{'' if ok else '  <-- MISMATCH'}")
        if not ok:
            failures.append(f"sum {name}")

    print(f"\nrow fingerprint: spark={spark['fingerprint']} ray={ray['fingerprint']}")
    if spark["fingerprint"] != ray["fingerprint"]:
        failures.append("row fingerprint")

    if failures:
        print("\nPARITY FAILED:", ", ".join(failures))
        sys.exit(1)
    print("\nPARITY OK: both outputs contain exactly the same rows")


if __name__ == "__main__":
    main()
