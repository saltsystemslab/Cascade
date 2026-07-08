# Compile-check container for GPU-Suffix-Array (caps_sa_gpu).
#
# Targets linux/amd64 for CUDA toolchain compatibility (runs under emulation on
# Apple Silicon). No GPU is required: this image only verifies that the project
# compiles. The hardcoded CUDA_ARCH in CMakeLists.txt (70-real, Volta/V100)
# matches our cluster, so no arch override is needed.

FROM --platform=linux/amd64 nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts (e.g. tzdata) during apt installs.
ENV DEBIAN_FRONTEND=noninteractive

# Standard build toolchain plus the project's host-side dependencies:
#   - build-essential / g++  : C/C++ host compiler
#   - cmake                  : build system (>= 3.17 required; 22.04 ships 3.22)
#   - libgomp1 / libomp-dev  : OpenMP runtime + headers (find_package(OpenMP))
#   - zlib1g-dev             : zlib for gzip support via kseq (find_package(ZLIB))
#   - git / wget / ca-certs  : fetching sources and CCCL headers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        cmake \
        ninja-build \
        libgomp1 \
        libomp-dev \
        zlib1g-dev \
        git \
        wget \
        ca-certificates \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install CCCL 2.7.0 headers (Thrust, CUB, libcu++) into /usr/local/include so
# they take precedence over the toolkit-bundled CCCL during compilation.
ARG CCCL_VERSION=2.7.0
RUN wget -qO /tmp/cccl.tar.gz \
        "https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_VERSION}.tar.gz" \
    && mkdir -p /tmp/cccl \
    && tar -xzf /tmp/cccl.tar.gz -C /tmp/cccl --strip-components=1 \
    && cp -r /tmp/cccl/thrust/thrust            /usr/local/include/ \
    && cp -r /tmp/cccl/cub/cub                  /usr/local/include/ \
    && cp -r /tmp/cccl/libcudacxx/include/cuda  /usr/local/include/ \
    && cp -r /tmp/cccl/libcudacxx/include/nv    /usr/local/include/ \
    && rm -rf /tmp/cccl /tmp/cccl.tar.gz

WORKDIR /workspace
