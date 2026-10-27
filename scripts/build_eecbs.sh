#!/usr/bin/env bash
# Build the official EECBS solver (Li, Ruml & Koenig, AAAI 2021) used for expert plans.
# Linux / macOS / WSL. Needs a C++14 compiler, CMake and Boost
# (Ubuntu/WSL: sudo apt install build-essential cmake git libboost-all-dev;
#  macOS:      brew install cmake boost).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=third_party/EECBS
mkdir -p third_party
if [ ! -d "$SRC" ]; then
  git clone --depth 1 https://github.com/Jiaoyang-Li/EECBS.git "$SRC"
fi

# Small compatibility patches for current toolchains (idempotent, no behaviour change):
#  - CMake >= 4 refuses cmake_minimum_required(VERSION 2.6)
#  - only Boost.program_options needs linking; Boost >= 1.89 no longer ships a
#    separate boost_system library and filesystem is unused
#  - newer GCC no longer pulls in <climits>/<cstdint> transitively (INT_MAX, uint64_t)
sed -i.bak -E 's/cmake_minimum_required *\( *VERSION *2\.6 *\)/cmake_minimum_required(VERSION 3.5)/' "$SRC/CMakeLists.txt"
sed -i.bak -E 's/find_package\( *Boost +REQUIRED +COMPONENTS +program_options +system +filesystem *\)/find_package(Boost REQUIRED COMPONENTS program_options)/' "$SRC/CMakeLists.txt"
if ! grep -q "<climits>" "$SRC/inc/common.h"; then
  sed -i.bak '1a #include <climits>\n#include <cstdint>' "$SRC/inc/common.h"
fi

rm -rf "$SRC/build"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$SRC/build" -j
echo
echo "Built: $(pwd)/$SRC/build/eecbs"
echo "It is found automatically from there (also from Windows Python via WSL),"
echo "or pass --eecbs_bin / set MAPF_NCO_EECBS."
