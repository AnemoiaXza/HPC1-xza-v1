#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <iostream>
#include "mpi.h"
#include <cblas.h>
#include <assert.h>
using namespace std;

void randMat(int rows, int cols, float *&Mat) {
    Mat = new float[rows * cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            Mat[i * cols + j] = 1.0;
}

int get_steps(int kernel, int step, int len) {
    if (kernel > len) return 0;
    return (len - kernel) / step + 1;
}

inline void img2col_conv_kernel(int leftAnchorX, int leftAnchorY, int rightAnchorX,
                                int rightAnchorY, const int xKernel, const int yKernel,
                                const int xStep, const int yStep,
                                float *&img, float *&kernel, float *&conv) {
    int imgRows = rightAnchorX - leftAnchorX,
        imgCols = rightAnchorY - leftAnchorY;
    int convRows = get_steps(xKernel, xStep, imgRows);
    int convCols = get_steps(yKernel, yStep, imgCols);
    float *flattenImg = new float[convRows * convCols * xKernel * yKernel];
    for (int i = leftAnchorX; i < rightAnchorX - xKernel; i += xStep) {
        for (int r = 0; r < xKernel; r++) {
            for (int j = leftAnchorY; j < rightAnchorY - yKernel; j += yStep) {
                int pos = (i - leftAnchorX)/xStep * convCols + (j - leftAnchorY) / yStep;
                memcpy(&flattenImg[pos * xKernel * yKernel + r * yKernel],
                       &img[(i + r) * imgCols + j],
                       sizeof(float) * yKernel);
            }
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, convRows * convCols, 1,
                xKernel * yKernel, 1.0, flattenImg, xKernel * yKernel, kernel, 1,
                0.0, conv, 1);
    delete flattenImg;
}

inline void naive_conv_kernel(int leftAnchorX, int leftAnchorY, int rightAnchorX,
                              int rightAnchorY, const int xKernel, const int yKernel,
                              const int xStep, const int yStep,
                              float *&img, float *&kernel, float *&conv) {
    int imgRows = rightAnchorX - leftAnchorX,
        imgCols = rightAnchorY - leftAnchorY;
    int convRows = get_steps(xKernel, xStep, imgRows);
    int convCols = get_steps(yKernel, yStep, imgCols);
    #pragma omp parallel for
    for (int i = leftAnchorX; i < rightAnchorX - xKernel; i += xStep) {
        for (int j = leftAnchorY; j < rightAnchorY - yKernel; j += yStep) {
            int pos = (i - leftAnchorX)/xStep * convCols + (j - leftAnchorY) / yStep;
            conv[pos] = 0.0;
            for (int r = i; r < i + xKernel; r++)
                for (int c = j; c < j + yKernel; c++) {
                    conv[pos] += img[r * imgCols + c] *
                                 kernel[(r - i) * yKernel + (c - j)];
                }
        }
    }
}

void mpi_convolution(int m, int n, int xKernel, int yKernel, int xStep,
                     int yStep, float *&img, float *&kernel, float *&conv,
                     int rank, int worldsize, bool img2col) {
    const int total_xsteps = get_steps(xKernel, xStep, m);
    const int total_ysteps = get_steps(yKernel, yStep, n);
    const int xsteps_per_proc = total_xsteps / worldsize;
    const int last_xsteps = total_xsteps - xsteps_per_proc * (worldsize - 1);
    int steps;
    if (rank == 0) {
        MPI_Request *sendRequest = new MPI_Request[worldsize];
        MPI_Status *status = new MPI_Status[worldsize];
        for (int i = 1; i < worldsize; i++) {
            steps = (i == worldsize - 1) ? last_xsteps : xsteps_per_proc;
            MPI_Isend(&img[i * xsteps_per_proc * xStep * n],
                      (steps * xStep + xKernel - xStep) * n, MPI_FLOAT, i, 0,
                      MPI_COMM_WORLD, &sendRequest[i]);
        }
        for (int i = 1; i < worldsize; i++) {
            MPI_Wait(&sendRequest[i], &status[i]);
        }
        delete sendRequest;
        delete status;
    } else {
        MPI_Status status;
        steps = (rank == worldsize - 1) ? last_xsteps : xsteps_per_proc;
        img = new float[(steps * xStep + xKernel - xStep) * n];
        MPI_Recv(img, (steps * xStep + xKernel - xStep) * n, MPI_FLOAT, 0, 0,
                 MPI_COMM_WORLD, &status);
        conv = new float[steps * total_ysteps];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    steps = (rank == worldsize - 1) ? last_xsteps : xsteps_per_proc;
    if (img2col)
        img2col_conv_kernel(0, 0, steps * xStep + xKernel - xStep, n, xKernel, yKernel,
                            xStep, yStep, img, kernel, conv);
    else
        naive_conv_kernel(0, 0, steps * xStep + xKernel - xStep, n, xKernel, yKernel,
                          xStep, yStep, img, kernel, conv);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Status status;
        for (int i = 1; i < worldsize; i++) {
            steps = (i == worldsize - 1) ? last_xsteps : xsteps_per_proc;
            MPI_Recv(&conv[i * xsteps_per_proc * total_ysteps],
                     steps * total_ysteps, MPI_FLOAT, i, 0, MPI_COMM_WORLD,
                     &status);
        }
    } else {
        MPI_Send(conv, steps * total_ysteps, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " M N enabled-img2col";
        exit(-1);
    }
    int rank;
    int worldSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int img2col = atoi(argv[3]);
    int xKernel = 3, yKernel = 3;
    int xStep = 1, yStep = 1;
    float *Img, *Conv;
    struct timeval start, stop;
    if (rank == 0) {
        randMat(m, n, Img);
        randMat(get_steps(xKernel, xStep, m), get_steps(yKernel, yStep, n),
                Conv);
    }
    float *Kernel = new float[xKernel*yKernel];
    for (int i = 0; i < xKernel*yKernel; i++) Kernel[i] = 1.0;
    gettimeofday(&start, NULL);
    mpi_convolution(m, n, xKernel, yKernel, xStep, yStep, Img, Kernel, Conv,
                    rank, worldSize, img2col);
    gettimeofday(&stop, NULL);
    if (rank == 0) {
        cout << "mpi convolution: "
             << (stop.tv_sec - start.tv_sec) * 1000.0 +
                    (stop.tv_usec - start.tv_usec) / 1000.0
             << " ms" << endl;
        for (int i = 0; i < min(10, m); i++) {
            for (int j = 0; j < min(10, n); j++)
                cout << Conv[i * n + j] << ' ';
            cout << endl;
        }
    }
    delete Img;
    delete Conv;
    MPI_Finalize();
}
