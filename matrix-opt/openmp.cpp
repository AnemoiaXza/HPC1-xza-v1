#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

using namespace std;

void randMat(int rows, int cols, float *&Mat) {
    Mat = new float[rows * cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            Mat[i * cols + j] = 1.0;
}

void openmp_sgemm(int m, int n, int k, float *leftMat, float *rightMat,
                  float *resultMat) {
    // transpose rightMat for cache-friendly access
    float *rightT = new float[k * n];
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < n; r++)
        for (int c = 0; c < k; c++)
            rightT[c * n + r] = rightMat[r * k + c];

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            float sum = 0.0;
            for (int i = 0; i < n; i++)
                sum += leftMat[row * n + i] * rightT[col * n + i];
            resultMat[row * k + col] = sum;
        }
    }
    delete[] rightT;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " M N K\n";
        exit(-1);
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    float *leftMat, *rightMat, *resMat;
    randMat(m, n, leftMat);
    randMat(n, k, rightMat);
    randMat(m, k, resMat);

    struct timeval start, stop;
    gettimeofday(&start, NULL);
    openmp_sgemm(m, n, k, leftMat, rightMat, resMat);
    gettimeofday(&stop, NULL);

    double t = (stop.tv_sec - start.tv_sec) * 1000.0 +
               (stop.tv_usec - start.tv_usec) / 1000.0;
    cout << "openmp matmul: " << t << " ms" << endl;

    // verify
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            if (int(resMat[i * k + j]) != n) {
                cout << resMat[i * k + j] << " error\n";
                exit(-1);
            }
    delete[] leftMat;
    delete[] rightMat;
    delete[] resMat;
    return 0;
}
