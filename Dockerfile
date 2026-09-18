#
# Multi-stage build: cargo build in the rust image, slim runtime.
# Build context is this directory:
#   docker build -t memory-server-rs .
# note: ort's prebuilt onnxruntime needs glibc >= 2.38 + GCC-13 libstdc++,
# so BOTH stages must be trixie-based (bookworm fails to link AND run)
FROM rust:1 AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:trixie-slim AS runtime
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/target/release/memory-server-rs /usr/local/bin/memory-server-rs
# ort loads the onnxruntime dylib from this cache at runtime (load-dynamic);
# without it the first embed() hangs trying to re-download it
COPY --from=build /root/.cache/ort.pyke.io /root/.cache/ort.pyke.io

ENV PORT=8080 \
    FASTEMBED_CACHE_PATH=/app/local_cache
# the local embedding model (bge-small-en-v1.5) is downloaded once on first
# startup and cached here; mount a volume on /app/local_cache to persist it.
RUN mkdir -p /app/local_cache
EXPOSE 8080

CMD ["memory-server-rs"]
