#!/usr/bin/env bash
# Build Ceres Solver 2.1.0 (CPU only) into ~/opt/ceres-2.1.0 without touching the system Ceres. Idempotent.
set -eo pipefail
PREFIX=$HOME/opt/ceres-2.1.0
[ -f "$PREFIX/lib/cmake/Ceres/CeresConfig.cmake" ] && { echo "Ceres 2.1.0 already installed in $PREFIX"; exit 0; }
mkdir -p ~/src && cd ~/src
[ -d ceres-solver ] || git clone --branch 2.1.0 --depth 1 https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DUSE_CUDA=OFF
make -j6
make install
echo "Ceres 2.1.0 installed in $PREFIX"
