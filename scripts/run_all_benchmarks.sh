#!/usr/bin/env bash
# Reproduce every benchmark artifact in bench/results/ end to end:
#   kernel microbenchmarks, ablation, build-thread scaling, filtered search,
#   cross-library comparison (engine vs hnswlib vs FAISS), plots and tables.
#
# Usage: scripts/run_all_benchmarks.sh [--synthetic] [--quick]
#   --synthetic  use generated stand-ins (synth-sift/glove/gist) instead of the
#                ann-benchmarks downloads (for machines without network access)
#   --quick      small subsets for a fast end-to-end check (~10 minutes)
# Environment: THREADS (default: nproc), PYTHON (default: python3),
#   SIFT_SUBSET / GLOVE_SUBSET / GIST_SUBSET (0 = full set).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
THREADS=${THREADS:-$(getconf _NPROCESSORS_ONLN)}
PREFIX=""
QUICK=0
for arg in "$@"; do
  case $arg in
    --synthetic) PREFIX="synth-" ;;
    --quick) QUICK=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done
if [[ $QUICK == 1 ]]; then
  SIFT_SUBSET=${SIFT_SUBSET:-100000}; GLOVE_SUBSET=${GLOVE_SUBSET:-100000}; GIST_SUBSET=${GIST_SUBSET:-50000}
  NQ=1000; REPS=1
else
  SIFT_SUBSET=${SIFT_SUBSET:-0}; GLOVE_SUBSET=${GLOVE_SUBSET:-0}; GIST_SUBSET=${GIST_SUBSET:-200000}
  NQ=0; REPS=3
fi
SIFT=${PREFIX}sift; GLOVE=${PREFIX}glove; GIST=${PREFIX}gist
RES=bench/results
mkdir -p "$RES" data

echo "== build (bench preset: -O3 -march=native) =="
cmake --preset bench >/dev/null
cmake --build --preset bench --target bench_main micro_bench
HNSW_NATIVE=ON $PYTHON -m pip install -q . hnswlib faiss-cpu matplotlib h5py
BM=build/bench/bench/bench_main

echo "== datasets =="
for d in sift glove gist; do
  if [[ ! -f data/${PREFIX}${d}_base.fvecs ]]; then
    if [[ -n $PREFIX ]]; then $PYTHON scripts/fetch_data.py --synthetic $d; else $PYTHON scripts/fetch_data.py $d; fi
  fi
done

echo "== environment =="
{
  echo "commit: $(git rev-parse HEAD)"
  echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
  if [[ $(uname -s) == Darwin ]]; then
    echo "cpu: $(sysctl -n machdep.cpu.brand_string) ($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo '?') performance + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0) efficiency cores)"
  else
    echo "cpu: $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p')"
  fi
  echo "cores: $(getconf _NPROCESSORS_ONLN) (threads used: $THREADS)"
  if [[ $(uname -s) == Darwin ]]; then
    echo "memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  else
    echo "memory: $(free -g 2>/dev/null | awk '/Mem:/{print $2 " GB"}')"
  fi
  echo "os: $(uname -srm)"
  echo "compiler: $(${CXX:-c++} --version | head -1)"
  echo "python: $($PYTHON --version)"
  $PYTHON -c "import importlib.metadata as m; print('hnswlib:', m.version('hnswlib')); print('faiss-cpu:', m.version('faiss-cpu')); print('numpy:', m.version('numpy'))"
  $PYTHON -c "import hnsw_engine; print('engine simd:', hnsw_engine.simd_isa())"
} | tee $RES/environment.txt

echo "== kernel microbenchmarks =="
build/bench/bench/micro_bench --benchmark_filter='BM_(L2|Dot)' --benchmark_min_time=0.2s \
  --benchmark_format=csv > $RES/micro_kernels.csv

echo "== brute-force sanity (recall must be 1.0) =="
rm -f $RES/bruteforce.csv
$BM --data data --name $SIFT --mode brute --search-threads 0 --reps 1 --nq 1000 --out $RES/bruteforce.csv

echo "== ablation ($SIFT) =="
rm -f $RES/ablation.csv
ABL_SUBSET=$([[ $QUICK == 1 ]] && echo 100000 || echo 0)
IDX=$(mktemp -d)/sift.idx
$BM --data data --name $SIFT --subset $ABL_SUBSET --nq 1 --ef 10 --reps 1 --build-threads $THREADS --save $IDX >/dev/null
abl() { $BM --data data --name $SIFT --subset $ABL_SUBSET --load $IDX --ef 64 --reps $REPS --out $RES/ablation.csv "$@"; }
abl --label "1 scalar kernels (no prefetch)" --isa scalar --no-prefetch --search-threads 1
case $(uname -m) in
  arm64|aarch64)  # Apple Silicon / ARM: NEON is the only SIMD ISA
    abl --label "2 +NEON kernels" --isa neon --no-prefetch --search-threads 1
    abl --label "3 +prefetch" --isa neon --search-threads 1 ;;
  *)
    abl --label "2 +AVX2 kernels" --isa avx2 --no-prefetch --search-threads 1
    abl --label "3 +prefetch" --isa avx2 --search-threads 1
    if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
      abl --label "4 +AVX-512 kernels" --isa avx512 --search-threads 1
    fi ;;
esac
abl --label "5 +${THREADS} search threads" --search-threads $THREADS
rm -f $IDX

echo "== build thread scaling ($SIFT) =="
rm -f $RES/scaling.csv
SCALE_SUBSET=$([[ $QUICK == 1 ]] && echo 50000 || echo 200000)
for t in 1 2 4 8 16; do
  (( t > THREADS )) && break
  $BM --data data --name $SIFT --subset $SCALE_SUBSET --nq 1000 --ef 64 --reps 1 --build-threads $t \
      --label "build-$t" --out $RES/scaling.csv
done

echo "== filtered search sweep =="
$PYTHON bench/python/filter_sweep.py --data data --name $SIFT --subset $SCALE_SUBSET --threads $THREADS

echo "== cross-library comparison =="
$PYTHON bench/python/compare.py --data data --name $SIFT --metric l2 --subset $SIFT_SUBSET --nq $NQ \
  --threads $THREADS --reps $REPS --build-1t
$PYTHON bench/python/compare.py --data data --name $GLOVE --metric cosine --subset $GLOVE_SUBSET --nq $NQ \
  --threads $THREADS --reps $REPS
$PYTHON bench/python/compare.py --data data --name $GIST --metric l2 --subset $GIST_SUBSET --nq $NQ \
  --threads $THREADS --reps $REPS

echo "== plots and tables =="
$PYTHON bench/python/plot.py
$PYTHON bench/python/report.py
echo "done: see $RES/summary.md"
