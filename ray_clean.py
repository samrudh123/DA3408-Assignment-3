import argparse
import csv
import glob
import os
import shutil
import time
from datetime import datetime

import pandas as pd
import ray

DATA_GLOB = "/data/yellow_tripdata_2026-*.parquet"
ZONES_PATH = "/data/taxi_zone_lookup.csv"
OUTPUT_PATH = "/output/ray_clean"
BENCHMARK_PATH = "/output/benchmarks.csv"
SHUFFLE_PARTITIONS = 64
TARGET_BLOCK_BYTES = 32 * 1024 * 1024
OBJECT_STORE_BUDGET_FRACTION = 4.0

START = pd.Timestamp(2026, 1, 1)
END = pd.Timestamp(2026, 5, 1)
MAX_DURATION_MIN = 360.0
MAX_SPEED_MPH = 80.0

RENAME = {
    "VendorID": "vendor_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
    "RatecodeID": "ratecode_id",
    "store_and_fwd_flag": "store_and_fwd_flag",
    "PULocationID": "pu_location_id",
    "DOLocationID": "do_location_id",
    "payment_type": "payment_type",
    "fare_amount": "fare_amount",
    "extra": "extra",
    "mta_tax": "mta_tax",
    "tip_amount": "tip_amount",
    "tolls_amount": "tolls_amount",
    "improvement_surcharge": "improvement_surcharge",
    "total_amount": "total_amount",
    "congestion_surcharge": "congestion_surcharge",
    "Airport_fee": "airport_fee",
    "cbd_congestion_fee": "cbd_congestion_fee",
}
CLEAN_COLUMNS = list(RENAME.values())
NULLABLE_INT_COLUMNS = {"passenger_count": "int64", "ratecode_id": "int64"}
ZONE_COLUMNS = [
    "pu_borough", "pu_zone", "pu_service_zone",
    "do_borough", "do_zone", "do_service_zone",
]
FINAL_COLUMNS = (
    CLEAN_COLUMNS
    + ["pickup_date", "pickup_hour"]
    + ZONE_COLUMNS
    + ["trip_duration_min", "avg_speed_mph", "hourly_avg_speed_mph"]
)


def trip_features(pickup, dropoff, distance):
    duration_min = (dropoff - pickup).total_seconds() / 60.0
    speed = distance / (duration_min / 60.0)
    return duration_min, speed


def trip_features_vectorized(pickup: pd.Series, dropoff: pd.Series, distance: pd.Series) -> pd.DataFrame:
    duration_min = (dropoff - pickup).dt.total_seconds() / 60.0
    speed = distance / (duration_min / 60.0)
    return pd.DataFrame({"trip_duration_min": duration_min, "avg_speed_mph": speed})


def load_zones(prefix):
    zones = pd.read_csv(ZONES_PATH, keep_default_na=False, dtype={"LocationID": "int32"})
    return zones.rename(columns={
        "LocationID": f"{prefix}_location_id",
        "Borough": f"{prefix}_borough",
        "Zone": f"{prefix}_zone",
        "service_zone": f"{prefix}_service_zone",
    })


def clean_batch(batch: pd.DataFrame) -> pd.DataFrame:
    batch = batch.dropna(how="any").rename(columns=RENAME).astype(NULLABLE_INT_COLUMNS)
    mask = (
        (batch["pickup_datetime"] >= START)
        & (batch["pickup_datetime"] < END)
        & (batch["dropoff_datetime"] > batch["pickup_datetime"])
        & (batch["trip_distance"] > 0)
        & (batch["trip_distance"] <= 100)
        & (batch["fare_amount"] > 0)
        & (batch["total_amount"] > 0)
        & (batch["passenger_count"] > 0)
    )
    batch = batch.loc[mask, CLEAN_COLUMNS]
    batch["row_key"] = pd.util.hash_pandas_object(batch, index=False).to_numpy()
    return batch


def drop_duplicates_block(batch: pd.DataFrame) -> pd.DataFrame:
    return batch.drop_duplicates(subset=CLEAN_COLUMNS)[CLEAN_COLUMNS]


def join_zones_batch(batch: pd.DataFrame, pu_zones: pd.DataFrame, do_zones: pd.DataFrame) -> pd.DataFrame:
    batch = batch.merge(pu_zones, on="pu_location_id", how="inner")
    batch = batch.merge(do_zones, on="do_location_id", how="inner")
    batch["pickup_date"] = batch["pickup_datetime"].dt.date
    batch["pickup_hour"] = batch["pickup_datetime"].dt.hour.astype("int32")
    return batch


def features_python_batch(batch: pd.DataFrame) -> pd.DataFrame:
    results = [
        trip_features(pickup, dropoff, distance)
        for pickup, dropoff, distance in zip(
            batch["pickup_datetime"].to_numpy(dtype="datetime64[us]").tolist(),
            batch["dropoff_datetime"].to_numpy(dtype="datetime64[us]").tolist(),
            batch["trip_distance"].tolist(),
        )
    ]
    batch["trip_duration_min"] = [r[0] for r in results]
    batch["avg_speed_mph"] = [r[1] for r in results]
    return batch


def features_pandas_batch(batch: pd.DataFrame) -> pd.DataFrame:
    features = trip_features_vectorized(
        batch["pickup_datetime"], batch["dropoff_datetime"], batch["trip_distance"]
    )
    batch["trip_duration_min"] = features["trip_duration_min"].to_numpy()
    batch["avg_speed_mph"] = features["avg_speed_mph"].to_numpy()
    return batch


def filter_outliers_batch(batch: pd.DataFrame) -> pd.DataFrame:
    mask = (batch["trip_duration_min"] <= MAX_DURATION_MIN) & (batch["avg_speed_mph"] <= MAX_SPEED_MPH)
    return batch.loc[mask]


def join_hourly_batch(batch: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    return batch.merge(hourly, on="pickup_hour", how="inner")[FINAL_COLUMNS]


def record_benchmark(row):
    new_file = not os.path.exists(BENCHMARK_PATH)
    with open(BENCHMARK_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--udf-mode", choices=["python", "pandas"], default="python")
    args = parser.parse_args()

    ray.init(address="auto")
    ctx = ray.data.DataContext.get_current()
    ctx.target_max_block_size = TARGET_BLOCK_BYTES
    ctx.override_object_store_memory_limit_fraction = OBJECT_STORE_BUDGET_FRACTION
    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)

    raw = ray.data.read_parquet(sorted(glob.glob(DATA_GLOB)))
    rows_raw = raw.count()

    t_start = time.perf_counter()

    cleaned = raw.map_batches(clean_batch, batch_format="pandas")
    deduped = cleaned.repartition(SHUFFLE_PARTITIONS, keys=["row_key"]).map_batches(
        drop_duplicates_block, batch_format="pandas", batch_size=None
    )
    stage1 = deduped.map_batches(
        join_zones_batch,
        batch_format="pandas",
        fn_kwargs={"pu_zones": load_zones("pu"), "do_zones": load_zones("do")},
    ).materialize()
    rows_clean = stage1.count()
    t_stage1 = time.perf_counter()

    udf = features_python_batch if args.udf_mode == "python" else features_pandas_batch
    stage2 = stage1.map_batches(udf, batch_format="pandas").materialize()
    t_stage2 = time.perf_counter()

    filtered = stage2.map_batches(filter_outliers_batch, batch_format="pandas")
    hourly = (
        filtered.groupby("pickup_hour")
        .mean("avg_speed_mph")
        .to_pandas()
        .rename(columns={"mean(avg_speed_mph)": "hourly_avg_speed_mph"})
        .astype({"pickup_hour": "int32"})
    )
    final = filtered.map_batches(join_hourly_batch, batch_format="pandas", fn_kwargs={"hourly": hourly})
    final.write_parquet(OUTPUT_PATH)
    t_end = time.perf_counter()

    rows_final = ray.data.read_parquet(OUTPUT_PATH).count()

    result = {
        "framework": "ray",
        "udf_mode": args.udf_mode,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "rows_raw": rows_raw,
        "rows_clean": rows_clean,
        "rows_final": rows_final,
        "ingest_clean_join_s": round(t_stage1 - t_start, 2),
        "udf_s": round(t_stage2 - t_stage1, 2),
        "features_export_s": round(t_end - t_stage2, 2),
        "total_s": round(t_end - t_start, 2),
    }
    record_benchmark(result)
    for key, value in result.items():
        print(f"{key:>20}: {value}")


if __name__ == "__main__":
    main()
