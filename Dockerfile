# Compile-check container for GPU-Suffix-Array (cascade).
#
# Targets linux/amd64 for CUDA toolchain compatibility (runs under emulation on
# Apple Silicon). No GPU is required: this image only verifies that the project
# compiles.

FROM --platform=linux/amd64 nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts (e.g. tzdata) during apt installs.
ENV DEBIAN_FRONTEND=noninteractive

# Standard build toolchain plus the project's host-side dependencies:
#   - build-essential / g++  : C/C++ host compiler
#   - cmake                  : build system (>= 3.22 required; 22.04 ships 3.22)
#   - libgomp1 / libomp-dev  : OpenMP runtime + headers (find_package(OpenMP))
#   - git / wget / ca-certs  : fetching sources and CCCL headers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        cmake \
        ninja-build \
        libgomp1 \
        libomp-dev \
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

# Modern clangd for in-container IDE indexing. Ubuntu 22.04's own clangd is 14,
# which predates CUDA 12.8 and chokes on the toolkit and CCCL 2.7 headers.
# libomp-N-dev is what puts omp.h into clang's own resource dir. clangd-N does not
# depend on it, and clang won't fall back to GCC's copy -- it borrows GCC's libstdc++
# paths but uses its own resource dir for builtin headers. Without this, every
# OpenMP-using TU fails to parse even though the GCC build is fine.
# Kept as the trailing layer so version bumps don't invalidate the CCCL layer.
ARG LLVM_VERSION=20
RUN apt-get update && apt-get install -y --no-install-recommends gnupg \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
         | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" \
         > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends clangd-${LLVM_VERSION} \
    && ln -sf /usr/bin/clangd-${LLVM_VERSION} /usr/bin/clangd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update && apt-get install -y --no-install-recommends \
         clangd-${LLVM_VERSION} libomp-${LLVM_VERSION}-dev \