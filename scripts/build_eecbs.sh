#!/usr/bin/env bash
# Build the official EECBS solver (Li, Ruml & Koenig, AAAI 2021) used for expert plans.
# Linux / macOS / WSL. Needs a C++14 compiler, CMake and Boost:
#   with sudo (Ubuntu/WSL): sudo apt install build-essential cmake git libboost-all-dev
#   without sudo (servers): conda install -c conda-forge boost cmake   (in your env, then re-run)
#   macOS:                  brew install cmake boost
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=third_party/EECBS

# ---- prerequisites -----------------------------------------------------------
missing=0
for tool in git cmake; do
  if ! command -v "$tool" >/dev/null 2>&1; then echo "ERROR: '$tool' not found"; missing=1; fi
done
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
  echo "ERROR: no C++ compiler (g++) found"; missing=1
fi
PREFIX_ARGS=()
BOOST_FOUND=0
for inc in /usr/include /usr/local/include /opt/homebrew/include "${CONDA_PREFIX:-/nonexistent}/include" "${BOOST_ROOT:-/nonexistent}/include"; do
  if [ -f "$inc/boost/program_options.hpp" ]; then BOOST_FOUND=1; fi
done
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/include/boost/program_options.hpp" ]; then
  echo "Using Boost from the active conda environment: $CONDA_PREFIX"
  PREFIX_ARGS=(-DCMAKE_PREFIX_PATH="$CONDA_PREFIX")
fi
if [ "$BOOST_FOUND" = 0 ]; then
  echo "ERROR: Boost headers not found."
  echo "  with sudo:    sudo apt install libboost-all-dev"
  echo "  without sudo: conda install -c conda-forge boost   (then re-run this script in that env)"
  missing=1
fi
if [ "$missing" = 1 ]; then exit 1; fi

# ---- source --------------------------------------------------------------------
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

# ---- build ---------------------------------------------------------------------
rm -rf "$SRC/build"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "${PREFIX_ARGS[@]}"
cmake --build "$SRC/build" -j

if [ ! -x "$SRC/build/eecbs" ]; then
  echo "ERROR: build finished but $SRC/build/eecbs is missing; see the messages above."
  exit 1
fi
echo
echo "Built: $(pwd)/$SRC/build/eecbs"
# EECBS exits with code 1 even for --help, so check that it prints its options instead.
HELP_OUT="$("$SRC/build/eecbs" --help 2>&1 || true)"
if ! grep -q "Allowed options" <<<"$HELP_OUT"; then
  echo "WARNING: the program does not start. Missing libraries:"
  (ldd "$SRC/build/eecbs" 2>/dev/null | grep "not found") || true
  echo "  try: export LD_LIBRARY_PATH=\${CONDA_PREFIX:-/usr/local}/lib:\$LD_LIBRARY_PATH"
fi
echo "It is found automatically from there (also from Windows Python via WSL),"
echo "or pass --eecbs_bin / set MAPF_NCO_EECBS."
