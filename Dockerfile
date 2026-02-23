FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS builder

RUN apt-get update --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && apt-get install --yes --quiet git wget gcc g++ cmake ninja-build make

# Install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet \
       python3.11 python3-pip python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install HMMER from source
RUN mkdir /hmmer_build /hmmer && \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build && \
    cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz && \
    cd hmmer-3.4 && ./configure --prefix /hmmer && \
    make -j8 && make install && \
    cd easel && make install && \
    rm -rf /hmmer_build

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --ignore-installed fastmcp loguru

# Clone and install AlphaFold3
RUN mkdir -p repo && \
    for attempt in 1 2 3; do \
      echo "Clone attempt $attempt/3"; \
      git clone --depth 1 https://github.com/charlesxu90/alphafold3.git repo/alphafold3 && break; \
      if [ $attempt -lt 3 ]; then sleep 5; fi; \
    done && \
    cd repo/alphafold3 && pip install --no-deps --no-cache-dir . && \
    build_data

# ---------- Runtime ----------
FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS runtime

RUN apt-get update --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet \
       python3.11 python3.11-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv, hmmer, and app from builder
COPY --from=builder /venv /venv
COPY --from=builder /hmmer /hmmer
COPY --from=builder /app/repo /app/repo

ENV PATH="/hmmer/bin:/venv/bin:$PATH"

WORKDIR /app
COPY src/ ./src/
RUN chmod -R a+r /app/src/
COPY configs/ ./configs/
RUN chmod -R a+r /app/configs/
RUN mkdir -p tmp/inputs tmp/outputs output jobs results && \
    chmod 777 /app /app/tmp /app/tmp/inputs /app/tmp/outputs /app/output /app/jobs /app/results

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

ENV NVIDIA_CUDA_END_OF_LIFE=0
ENTRYPOINT []
CMD ["python", "src/server.py"]
