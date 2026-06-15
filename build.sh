#!/usr/bin/env bash
#
# Compile-check the project inside a Docker container (no GPU required).
#
# Builds the Docker image on first use, mounts this project directory into the
# container, and runs cmake + make there, writing artifacts to ./build.
#
# Usage:
#   ./build.sh [extra cmake flags...]
#   ./build.sh -DCMAKE_BUILD_TYPE=Debug -DTESTS=ON
#   REBUILD_IMAGE=1 ./build.sh        # force a rebuild of the Docker image
#
# Any arguments are forwarded verbatim to the cmake configure step.

set -euo pipefail

IMAGE_NAME="gpu-suffix-array-build"
PLATFORM="linux/amd64"

# Resolve the project root (directory containing this script) so build.sh works
# regardless of the caller's current working directory.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the image if it doesn't exist yet (or if REBUILD_IMAGE is set).
if [[ "${REBUILD_IMAGE:-0}" == "1" ]] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo ">> Building Docker image '$IMAGE_NAME' ($PLATFORM)..."
    docker build --platform "$PLATFORM" -t "$IMAGE_NAME" "$PROJECT_DIR"
else
    echo ">> Using existing Docker image '$IMAGE_NAME' (set REBUILD_IMAGE=1 to rebuild)."
fi

echo ">> Compiling inside container (artifacts -> ./build)..."
# Any arguments to this script ("$@") are forwarded verbatim to the cmake
# configure step. "$@" is used directly (not copied into a named array) so it
# stays safe under `set -u` when empty on macOS's bash 3.2.
docker run --rm \
    --platform "$PLATFORM" \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c '
        set -euo pipefail
        cmake -S . -B build "$@"
        cmake --build build -j"$(nproc)"
    ' bash "$@"

echo ">> Build finished. Artifacts are in ./build"
