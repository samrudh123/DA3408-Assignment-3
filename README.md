# DA3408 Assignment 3: Spark vs Ray, the Data Engineering Duel

The same data-preprocessing pipeline is implemented in **Apache Spark** (`spark_clean.py`) and **Ray Data** (`ray_clean.py`). Each runs on its own two-worker cluster in Docker, on the NYC TLC Yellow Taxi trip records for January to April 2026 (14.9M rows, about 2.25 GB in memory). The two are compared on execution time, resource usage and Python UDF overhead.

The results, charts and the final verdict ("winner") are in **[A3_Report.pdf](A3_Report.pdf)**. AI usage is declared in **[AI_DISCLOSURE.md](AI_DISCLOSURE.md)**.

---

## Repository layout

| Path | Purpose |
|---|---|
| `Dockerfile` | One image for every Spark and Ray node: Python 3.12, OpenJDK 17, pyspark 4.2.0, ray 2.58.0, running as UID 1000 |
| `docker-compose-spark.yml` | Spark standalone cluster: 1 master and 2 workers on the `sparknet` network |
| `docker-compose-ray.yml` | Ray cluster: 1 head and 2 workers on the `raynet` network |
| `spark_clean.py` | Spark pipeline |
| `ray_clean.py` | Ray Data pipeline (same logic) |
| `compare.py` | Parity check: proves both outputs contain exactly the same rows |
| `screenshots/` | Spark and Ray UI screenshots embedded in the report |
| `data/` | Input data (not committed; see below) |
| `output/` | Pipeline outputs, `benchmarks.csv` and resource logs (not committed) |
| `scratch/` | Spark and Ray spill directories (not committed) |

---

## Pipeline

Both scripts run the same steps, and `compare.py` checks that the results are identical.

| Step | Logic | Spark | Ray Data |
|---|---|---|---|
| **Ingestion** | Read `data/yellow_tripdata_2026-0[1-4].parquet` | `spark.read.parquet` | `ray.data.read_parquet` |
| **Cleansing** | Drop rows with any null; rename columns to snake_case; keep pickups in Jan–Apr 2026 with dropoff after pickup, 0 < distance ≤ 100, fare > 0, total > 0, passengers > 0; remove exact duplicate rows | `na.drop`, `filter`, `dropDuplicates()` | pandas `map_batches`; dedup via a 64-bit row hash, `repartition(keys=["row_key"])`, then `drop_duplicates` per block |
| **Heavy join** | Join pickup and dropoff zone IDs to `taxi_zone_lookup.csv` (borough, zone, service zone); derive `pickup_date` and `pickup_hour` | broadcast hash join | broadcast join inside `map_batches` |
| **Python UDF** | `trip_features()` gives `trip_duration_min` and `avg_speed_mph` | `--udf-mode python`: `udf(useArrow=False)`; `--udf-mode pandas`: `pandas_udf` | `--udf-mode python`: the same function per row in native Python; `--udf-mode pandas`: vectorised |
| **Features** | Drop trips over 6 h or 80 mph; join back the average speed per pickup hour (`hourly_avg_speed_mph`) | `groupBy` + broadcast join | `groupby().mean()` + broadcast join |
| **Export** | 31-column parquet, ready for ML | `output/spark_clean/` | `output/ray_clean/` |

Each stage is materialised before the next one starts (`persist().count()` in Spark, `materialize()` in Ray). That makes the UDF time measurable on its own. Every run appends one row to `output/benchmarks.csv`, with these columns:
- row counts (raw, clean, final)
- `ingest_clean_join_s`
- `udf_s`
- `features_export_s`
- `total_s`

---

## Cluster design

| Node | Spark | Ray | Limits |
|---|---|---|---|
| Coordinator | `spark-master` (7077, UI 8080, driver UI 4040) | `ray-head` (`--num-cpus=0`, GCS 6379, dashboard 8265) | 2 CPU, 2 GB |
| Worker 1 | `spark-worker-1` (`--cores 4 --memory 2g`) | `ray-worker-1` (`--num-cpus=4`, 1 GB object store) | 4 CPU, 4 GB |
| Worker 2 | `spark-worker-2` (`--cores 4 --memory 2g`) | `ray-worker-2` (`--num-cpus=4`, 1 GB object store) | 4 CPU, 4 GB |
| Driver | `spark-submit` inside `spark-master` | `python` inside `ray-head` | – |
| Spill | `SPARK_LOCAL_DIRS=/scratch/<node>` | `--object-spilling-directory=/scratch/<node>` | host disk |

- `./data`, `./output`, `./scratch` and the repo itself are mounted at the same paths on every node.
- Run **one cluster at a time**: together they would exceed a 14 GB machine.

---

## Prerequisites

- Linux with Docker Engine and Docker Compose v2
- About 10 GB of free RAM while a cluster runs
- About 5 GB of free disk where `scratch/` lives (spill files)
- To build the report: a Python 3.12 environment with `pandas`, `pyarrow`, `numpy`, `matplotlib` and `reportlab`

### Get the data

Download into `data/` from the [NYC TLC trip record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

```
mkdir -p data output scratch
cd data
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-02.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-03.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet
wget https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
cd ..
```

Create `output/` and `scratch/` yourself before starting a cluster. Otherwise Docker creates them as root and the containers can't write to them.

---

## Running

### 1. Build the image
```
docker build -t da3408-a3:latest .
```

### 2. Spark
```
docker compose -f docker-compose-spark.yml up -d
docker logs spark-worker-1 2>&1 | grep -i registered
docker logs spark-worker-2 2>&1 | grep -i registered
```

Check `http://localhost:8080`: there should be 2 ALIVE workers with 4 cores each. Then run the pipeline:
```
docker exec spark-master spark-submit --master spark://spark-master:7077 --executor-cores 4 --executor-memory 2g /app/spark_clean.py --udf-mode python
docker exec spark-master spark-submit --master spark://spark-master:7077 --executor-cores 4 --executor-memory 2g /app/spark_clean.py --udf-mode pandas
docker compose -f docker-compose-spark.yml down
```

### 3. Ray
```
docker compose -f docker-compose-ray.yml up -d
docker exec ray-head ray status
```

Expect 3 nodes and 8 CPUs; the dashboard is at `http://localhost:8265`. Then run the pipeline:
```
docker exec ray-head python /app/ray_clean.py --udf-mode python
docker exec ray-head python /app/ray_clean.py --udf-mode pandas
docker compose -f docker-compose-ray.yml down
```

### 4. Parity check
```
docker run --rm -v "$PWD":/app -w /app da3408-a3:latest python compare.py
```

It ends with `PARITY OK: both outputs contain exactly the same rows`, and exits with code 1 on any mismatch.

### 5. Benchmarking

Run each of the four configurations (Spark/Ray × python/pandas) at least 3 times.

While a cluster runs, record resource usage in a second terminal (`Ctrl+C` to stop). Use `resource_usage_ray.csv` for Ray:
```
while true; do docker stats --no-stream --format "{{.Name}},{{.CPUPerc}},{{.MemUsage}}" | sed "s/^/$(date +%T),/" >> output/resource_usage_spark.csv; done
```

### 6. Screenshots

The UI screenshots used in the report:
- `screenshots/spark_master_ui.png`: Spark Master UI (port 8080) with 2 ALIVE workers
- `screenshots/spark_job_ui.png`: Spark application UI (port 4040) during a run
- `screenshots/ray_dashboard.png`: Ray Dashboard (port 8265) cluster view
- `screenshots/ray_job.png`: Ray Dashboard during a run

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Failed to create directory .../pyspark/work` | The worker's work dir isn't writable. The image sets `SPARK_WORKER_DIR=/tmp/spark-work`, so rebuild the image. |
| A worker shows all host cores | `--cores` (Spark) or `--num-cpus` (Ray) is missing |
| `OutOfDiskError: Local disk is full` (Ray) | Spill files went to the container's root disk. Check that `./scratch` is mounted and `--object-spilling-directory` is set. |
| `(raylet) Task ... failed due to oom ... will be retried` | Ray's memory monitor killed a task near the 4 GB limit and retried it. The result is still correct. |
| Ray stage runs with `Tasks: 1 [backpressured]` | Ray Data's object-store budget is exhausted. `ray_clean.py` sets `override_object_store_memory_limit_fraction=4.0` so excess data spills to disk instead. |
| `Permission denied` on `output/` | Docker created the folder as root. Fix with `sudo chown -R $USER output scratch`. |
| Job hangs at "Initial job has not accepted any resources" | `--executor-memory` is larger than the worker's `--memory` |
