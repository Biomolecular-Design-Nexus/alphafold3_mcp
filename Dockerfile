FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS builder

RUN apt-get update --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && apt-get install --yes --quiet git wget gcc g++ cmake ninja-build make \
       patch flex bison curl zlib1g-dev libboost-all-dev

# Install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet \
       python3.11 python3-pip python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

# Clone AlphaFold3 first (needed for jackhmmer patch)
RUN mkdir -p repo && \
    for attempt in 1 2 3; do \
      echo "Clone attempt $attempt/3"; \
      git clone --depth 1 https://github.com/charlesxu90/alphafold3.git repo/alphafold3 && break; \
      if [ $attempt -lt 3 ]; then sleep 5; fi; \
    done

# Install HMMER 3.4 from source (with jackhmmer patch if available)
RUN mkdir /hmmer_build /hmmer && \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build && \
    cd /hmmer_build && \
    echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check && \
    tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz && \
    if [ -f /app/repo/alphafold3/docker/jackhmmer_seq_limit.patch ]; then \
      cp /app/repo/alphafold3/docker/jackhmmer_seq_limit.patch /hmmer_build/ && \
      cd /hmmer_build && patch -p0 < jackhmmer_seq_limit.patch; \
    fi && \
    cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer && \
    make -j8 && make install && \
    cd easel && make install && \
    rm -rf /hmmer_build

# Install maxit for CIF to PDB conversion
RUN mkdir -p /tmp/maxit_build && cd /tmp/maxit_build && \
    wget -q https://sw-tools.rcsb.org/apps/MAXIT/maxit-v11.300-prod-src.tar.gz && \
    tar xzf maxit-v11.300-prod-src.tar.gz && \
    cd maxit-v11.300-prod-src && \
    make && make binary && \
    mv /tmp/maxit_build/maxit-v11.300-prod-src /opt/maxit && \
    rm -rf /tmp/maxit_build

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --ignore-installed fastmcp loguru

# Install build tools for C++ extensions, then install AlphaFold3 and build data
RUN pip install --no-cache-dir scikit-build-core pybind11 cmake ninja numpy && \
    cd repo/alphafold3 && pip install --no-cache-dir . && \
    build_data && \
    cp /venv/lib/python3.11/site-packages/alphafold3/cpp.cpython-311-x86_64-linux-gnu.so \
       /app/repo/alphafold3/src/alphafold3/ && \
    cp /venv/lib/python3.11/site-packages/alphafold3/constants/converters/*.pickle \
       /app/repo/alphafold3/src/alphafold3/constants/converters/

# Fix: Copy OUTPUT_TERMS_OF_USE.md to source package dir (run_alphafold.py resolves it via cpp.__file__)
RUN cp /app/repo/alphafold3/OUTPUT_TERMS_OF_USE.md \
       /app/repo/alphafold3/src/alphafold3/OUTPUT_TERMS_OF_USE.md

# Fix: Replace ETKDGv3 with srETKDGv3 for macrocyclic peptide conformer generation
# ETKDGv3 hangs indefinitely on large macrocycles; srETKDGv3 handles them correctly
COPY patches/fix_srETKDGv3.py /tmp/fix_srETKDGv3.py
RUN python /tmp/fix_srETKDGv3.py && rm /tmp/fix_srETKDGv3.py

# ---------- Runtime ----------
FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS runtime

RUN apt-get update --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet \
       python3.11 python3.11-venv libgomp1 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Copy venv, hmmer, maxit, and app from builder
COPY --from=builder /venv /venv
COPY --from=builder /hmmer /hmmer
COPY --from=builder /opt/maxit /opt/maxit
COPY --from=builder /app/repo /app/repo

ENV PATH="/hmmer/bin:/opt/maxit/bin:/venv/bin:$PATH"
ENV RCSBROOT=/opt/maxit

WORKDIR /app
COPY src/ ./src/
RUN chmod -R a+r /app/src/
COPY configs/ ./configs/
RUN chmod -R a+r /app/configs/
COPY scripts/ ./scripts/
RUN chmod -R a+r /app/scripts/
RUN mkdir -p tmp/inputs tmp/outputs output jobs results && \
    chmod 777 /app /app/tmp /app/tmp/inputs /app/tmp/outputs /app/output /app/jobs /app/results

ENV PYTHONPATH=/app:/app/repo/alphafold3/src
ENV PYTHONUNBUFFERED=1
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

ENV NVIDIA_CUDA_END_OF_LIFE=0
ENTRYPOINT []
CMD ["python", "src/server.py"]
