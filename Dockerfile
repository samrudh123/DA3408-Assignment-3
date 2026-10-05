FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN pip install --no-cache-dir \
        pyspark==4.2.0 \
        "ray[data,default]==2.58.0" \
        pyarrow==25.0.0 \
        pandas==2.3.3 \
        numpy==2.5.2 \
        psutil==7.2.2

ENV SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark
ENV PATH="${SPARK_HOME}/bin:${PATH}"
ENV PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3 \
    SPARK_WORKER_DIR=/tmp/spark-work \
    PYTHONUNBUFFERED=1

RUN groupadd -g 1000 app && useradd -m -u 1000 -g 1000 app
USER app
WORKDIR /app
