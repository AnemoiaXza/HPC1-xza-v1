#!/bin/bash
# Benchmark: compare serial, openmp, mpi-original (from ../matrix), mpi-shm

export LD_LIBRARY_PATH=$HOME/openblas-install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/mpich-install/bin:$PATH

DIR="$(cd "$(dirname "$0")" && pwd)"
MATRIX_DIR="$DIR/../matrix"
OUTPUT="$DIR/bench_result.csv"

# matrix sizes to test
SIZES="512 1024 2048 4096"
# thread/proc counts
THREADS="1 2 4 8 16"
PROCS="1 4 9 16"

echo "method,size,workers,time_ms" > "$OUTPUT"

for sz in $SIZES; do
    echo "--- size=$sz ---"

    # serial (baseline)
    t=$($DIR/serial $sz $sz $sz 2>&1 | grep 'serial' | awk '{print $3}')
    echo "serial,$sz,1,$t" >> "$OUTPUT"
    echo "  serial: $t ms"

    # openmp with various thread counts
    for th in $THREADS; do
        t=$(OMP_NUM_THREADS=$th $DIR/openmp $sz $sz $sz 2>&1 | grep 'openmp' | awk '{print $3}')
        echo "openmp,$sz,$th,$t" >> "$OUTPUT"
        echo "  openmp th=$th: $t ms"
    done

    # mpi original (from matrix/)
    for np in $PROCS; do
        t=$(mpirun -np $np $MATRIX_DIR/gemm $sz $sz $sz 0 2>&1 | grep 'mpi matmul' | awk '{print $3}')
        echo "mpi-original,$sz,$np,$t" >> "$OUTPUT"
        echo "  mpi-orig np=$np: $t ms"
    done

    # mpi-shm
    for np in $PROCS; do
        t=$(mpirun -np $np $DIR/mpi-shm $sz $sz $sz 2>&1 | grep 'mpi-shm' | awk '{print $3}')
        echo "mpi-shm,$sz,$np,$t" >> "$OUTPUT"
        echo "  mpi-shm np=$np: $t ms"
    done
done

echo "Done. Results in $OUTPUT"
