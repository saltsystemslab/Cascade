#!/usr/bin/env bash
#
# Launch clangd inside the build container so it sees the toolchain the build
# actually uses: CUDA 12.8, CCCL 2.7, libstdc++, omp.h.
#
# The project is mounted at its host path, identically to build.sh, so the paths
# in build/compile_commands.json are valid on both sides and no --path-mappings
# translation is needed.
#
# Wired up through .vscode/settings.json -> "clangd.path".
#
# NOTE: --entrypoint is load-bearing, not decorative. The nvidia/cuda base image's
# ENTRYPOINT prints a CUDA banner to stdout before exec'ing the command, and stdout
# here carries the LSP stream -- the banner corrupts the protocol and the client
# drops the connection. Bypassing the entrypoint is required, not merely tidier.
# Nothing in this script may write to stdout either.

set -euo pipefail

IMAGE_NAME="gpu-suffix-array-build"
PLATFORM="linux/amd64"

# Resolve the project root by walking up from this script's own directory until
# the root markers appear. Depth-independent on purpose.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$PROJECT_DIR/CMakeLists.txt" || ! -f "$PROJECT_DIR/build.sh" ]]; do
    if [[ "$PROJECT_DIR" == "/" ]]; then
        echo "clangd-docker.sh: cannot locate project root above ${BASH_SOURCE[0]}" >&2
        exit 1
    fi
    PROJECT_DIR="$(dirname "$PROJECT_DIR")"
done

exec docker run --rm -i \
    --platform "$PLATFORM" \
    --entrypoint /usr/bin/clangd \
    -v "$PROJECT_DIR":"$PROJECT_DIR" \
    -w "$PROJECT_DIR" \
    "$IMAGE_NAME" \
    "$@"
