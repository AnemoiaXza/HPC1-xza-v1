#!/bin/bash
# Benchmark three original MPI experiments: gemm, conv, pooling

export LD_LIBRARY_PATH=$HOME/openblas-install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/mpich-install/bin:$PATH

DIR="$(cd "$(dirname "$0")" && pwd)"
MATRIX_DIR="$DIR/../matrix"
OUTPUT="$DIR/bench_3exp.csv"

SIZES="512 1024 2048"  # conv/pooling sizes (4096 too slow for naive conv)
PROCS="1 2 4 8"

echo "experiment,size,workers,time_ms" > "$OUTPUT"

for sz in $SIZES; do
    echo "--- size=$sz ---"

    # gemm
    for np in $PROCS; do
        t=$(mpirun -np $np $MATRIX_DIR/gemm $sz $sz $sz 0 2>&1 | grep 'mpi matmul' | awk '{print $3}')
        echo "gemm,$sz,$np,$t" >> "$OUTPUT"
        echo "  gemm np=$np: $t ms"
    done

    # conv (naive: mode=0)
    for np in $PROCS; do
        t=$(mpirun -np $np $MATRIX_DIR/conv $sz $sz 0 2>&1 | grep 'mpi convolution' | awk '{print $3}')
        echo "conv-naive,$sz,$np,$t" >> "$OUTPUT"
        echo "  conv-naive np=$np: $t ms"
    done

    # conv (img2col: mode=1)
    for np in $PROCS; do
        t=$(mpirun -np $np $MATRIX_DIR/conv $sz $sz 1 2>&1 | grep 'mpi convolution' | awk '{print $3}')
        echo "conv-img2col,$sz,$np,$t" >> "$OUTPUT"
        echo "  conv-img2col np=$np: $t ms"
    done

    # pooling
    for np in $PROCS; do
        t=$(mpirun -np $np $MATRIX_DIR/pooling $sz $sz 2>&1 | grep 'mpi pooling' | awk '{print $3}')
        echo "pooling,$sz,$np,$t" >> "$OUTPUT"
        echo "  pooling np=$np: $t us"
    done
done

# Clean bad rows
sed -i '/,-/d' "$OUTPUT"
echo "Done. Results in $OUTPUT"
