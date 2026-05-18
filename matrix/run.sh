#!/bin/bash

export LD_LIBRARY_PATH=$HOME/openblas-install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/mpich-install/bin:$PATH

app=${1}
np=${2:-4}

if [ "$app" = "gemm" ]; then
    mpirun -np ${np} ./gemm 1024 1024 1024 0
fi

if [ "$app" = "conv" ]; then
    mpirun -np ${np} ./conv 4096 4096 ${3:-0}
fi

if [ "$app" = "pooling" ]; then
    mpirun -np ${np} ./pooling 1024 1024
fi
