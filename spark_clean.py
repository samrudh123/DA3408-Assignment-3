import argparse
import csv
import glob
import os
import time
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession, functions as F, types as T

DATA_GLOB = "/data/yellow_tripdata_2026-*.parquet"
ZONES_PATH = "/data/taxi_zone_lookup.csv"
OUTPUT_PATH = "/output/spark_clean"
BENCHMARK_PATH = "/output/benchmarks.csv"
SHUFFLE_PARTITIONS = 64

START = datetime(2026, 1, 1)
END = datetime(2026, 5, 1)
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

FEATURE_TYPE = T.StructType([
    T.StructField("trip_duration_min", T.DoubleType()),
    T.StructField("avg_speed_mph", T.DoubleType()),
])


def trip_features(pickup, dropoff, distance):
    duration_min = (dropoff - pickup).total_seconds() / 60.0
    speed = distance / (duration_min / 60.0)
    return duration_min, speed


def trip_features_vectorized(pickup: pd.Series, dropoff: pd.Series, distance: pd.Series) -> pd.DataFrame:
    duration_min = (dropoff - pickup).dt.total_seconds() / 60.0
    speed = distance / (duration_min / 60.0)
    return pd.DataFrame({"trip_duration_min": duration_min, "avg_speed_mph": speed})


def build_spark():
    spark = (
        SparkSession.builder.appName("spark_clean")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_zones(spark, prefix):
    schema = "LocationID INT, Borough STRING, Zone STRING, service_zone STRING"
    zones = spark.read.csv(ZONES_PATH, header=True, schema=schema)
    return zones.select(
        F.col("LocationID").alias(f"{prefix}_location_id"),
        F.col("Borough").alias(f"{prefix}_borough"),
        F.col("Zone").alias(f"{prefix}_zone"),
        F.col("service_zone").alias(f"{prefix}_service_zone"),
    )


def clean(raw):
    df = raw.na.drop(how="any")
    df = df.select([F.col(src).alias(dst) for src, dst in RENAME.items()])
    return df.filter(
        (F.col("pickup_datetime") >= F.lit(START))
        & (F.col("pickup_datetime") < F.lit(END))
        & (F.col("dropoff_datetime") > F.col("pickup_datetime"))
        & (F.col("trip_distance") > 0)
        & (F.col("trip_distance") <= 100)
        & (F.col("fare_amount") > 0)
        & (F.col("total_amount") > 0)
        & (F.col("passenger_count") > 0)
    )


def join_zones(df, pu_zones, do_zones):
    df = df.join(F.broadcast(pu_zones), on="pu_location_id", how="inner")
    df = df.join(F.broadcast(do_zones), on="do_location_id", how="inner")
    return df.withColumn("pickup_date", F.to_date("pickup_datetime")).withColumn(
        "pickup_hour", F.hour("pickup_datetime")
    )


def add_features(df, udf_mode):
    if udf_mode == "python":
        feature_udf = F.udf(trip_features, FEATURE_TYPE, useArrow=False)
    else:
        feature_udf = F.pandas_udf(trip_features_vectorized, FEATURE_TYPE)
    df = df.withColumn("features", feature_udf("pickup_datetime", "dropoff_datetime", "trip_distance"))
    return df.select("*", "features.trip_duration_min", "features.avg_speed_mph").drop("features")


def add_hourly_speed(df):
    df = df.filter(
        (F.col("trip_duration_min") <= MAX_DURATION_MIN) & (F.col("avg_speed_mph") <= MAX_SPEED_MPH)
    )
    hourly = df.groupBy("pickup_hour").agg(F.avg("avg_speed_mph").alias("hourly_avg_speed_mph"))
    return df.join(F.broadcast(hourly), on="pickup_hour", how="inner").select(FINAL_COLUMNS)


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

    spark = build_spark()
    raw = spark.read.parquet(*sorted(glob.glob(DATA_GLOB)))
    rows_raw = raw.count()

    t_start = time.perf_counter()

    stage1 = join_zones(clean(raw).dropDuplicates(), load_zones(spark, "pu"), load_zones(spark, "do")).persist()
    rows_clean = stage1.count()
    t_stage1 = time.perf_counter()

    stage2 = add_features(stage1, args.udf_mode).persist()
    stage2.count()
    t_stage2 = time.perf_counter()
    stage1.unpersist()

    final = add_hourly_speed(stage2)
    final.write.mode("overwrite").parquet(OUTPUT_PATH)
    t_end = time.perf_counter()

    rows_final = spark.read.parquet(OUTPUT_PATH).count()
    stage2.unpersist()

    result = {
        "framework": "spark",
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

    spark.stop()


if __name__ == "__main__":
    main()
