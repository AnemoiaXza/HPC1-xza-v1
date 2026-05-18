#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include "mpi.h"
using namespace std;

void randMat(int rows, int cols, float *&Mat) {
    Mat = new float[rows * cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            Mat[i * cols + j] = 1.0;
}

inline void pooling_kernel(int leftAnchorX, int leftAnchorY, int rightAnchorX,
                           int rightAnchorY, int xStride, int yStride,
                           int from_m, int from_n, int to_m, int to_n,
                           float *&from, float *&to) {
    if ((rightAnchorX - leftAnchorX) % xStride != 0 ||
        (rightAnchorY - leftAnchorY) % yStride != 0)
        exit(-1);
    if (leftAnchorX % xStride != 0 || leftAnchorY % yStride != 0) exit(-2);
    for (int i = leftAnchorX; i < rightAnchorX; i += xStride) {
        for (int j = leftAnchorY; j < rightAnchorY; j += yStride) {
            float temp = to[i / xStride * to_n + j / yStride];
            for (int r = i; r < i + xStride; r++)
                for (int c = j; c < j + yStride; c++) {
                    temp = max(temp, from[r * from_n + c]);
                }
            to[i / xStride * to_n + j / yStride] = temp;
        }
    }
}

void mpi_pooling(int m, int n, int xstride, int ystride, float *&mat,
                 float *&res, int rank, int worldsize) {
    if (m % xstride || n % ystride) {
        cout << "matrix size and stride do not match \n";
        return;
    }
    const int xstrides_per_proc = (m / xstride) / worldsize;
    int strides;
    if (rank == 0) {
        MPI_Request *sendRequest = new MPI_Request[worldsize];
        MPI_Status *status = new MPI_Status[worldsize];
        for (int i = 1; i < worldsize; i++) {
            strides = (i < worldsize - 1)
                          ? xstrides_per_proc
                          : (m / xstride) - xstrides_per_proc * (worldsize - 1);
            MPI_Isend(&mat[i * xstrides_per_proc * xstride * n],
                      strides * xstride * n, MPI_FLOAT, i, 0, MPI_COMM_WORLD,
                      &sendRequest[i]);
        }
        for (int i = 1; i < worldsize; i++) {
            MPI_Wait(&sendRequest[i], &status[i]);
        }
        delete sendRequest;
        delete status;
    } else {
        MPI_Status status;
        strides = (rank < worldsize - 1)
                      ? xstrides_per_proc
                      : (m / xstride) - xstrides_per_proc * (worldsize - 1);
        mat = new float[strides * xstride * n];
        MPI_Recv(mat, strides * xstride * n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
                 &status);
        res = new float[strides * (n / ystride)];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    strides = (rank < worldsize - 1)
                  ? xstrides_per_proc
                  : (m / xstride) - xstrides_per_proc * (worldsize - 1);
    pooling_kernel(0, 0, strides * xstride, n, xstride, ystride,
                   strides * xstride, n, strides, n / ystride, mat, res);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Status status;
        for (int i = 1; i < worldsize; i++) {
            strides = (i < worldsize - 1)
                          ? xstrides_per_proc
                          : (m / xstride) - xstrides_per_proc * (worldsize - 1);
            MPI_Recv(&res[i * xstrides_per_proc * (n / ystride)],
                     strides * (n / ystride), MPI_FLOAT, i, 0, MPI_COMM_WORLD,
                     &status);
        }
    } else {
        MPI_Send(res, strides * (n / ystride), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " M N";
        exit(-1);
    }
    int rank;
    int worldSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int xstride = 4, ystride = 4;
    float *Mat, *resMat;
    struct timeval start, stop;
    if (rank == 0) {
        randMat(m, n, Mat);
        randMat(m / xstride, n / ystride, resMat);
    }
    gettimeofday(&start, NULL);
    mpi_pooling(m, n, xstride, ystride, Mat, resMat, rank, worldSize);
    gettimeofday(&stop, NULL);
    if (rank == 0) {
        cout << "mpi pooling: "
             << (stop.tv_sec - start.tv_sec) * 1000 * 1000L +
                    (stop.tv_usec - start.tv_usec)
             << endl;
        for (int i = 0; i < min(10, m); i++) {
            for (int j = 0; j < min(10, n); j++)
                cout << Mat[i * n + j] << ' ';
            cout << endl;
        }
    }
    delete Mat;
    delete resMat;
    MPI_Finalize();
}
