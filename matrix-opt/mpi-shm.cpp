#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 4) {
        if (argc > 0) cout << "Usage: " << argv[0] << " M N K\n";
        exit(-1);
    }

    int rank, worldSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // single shared memory window: leftMat + rightMat + resMat
    size_t total_size = (m * n + n * k + m * k) * sizeof(float);
    int disp_unit = sizeof(float);
    MPI_Win win;
    float *base = NULL;

    MPI_Win_allocate_shared((rank == 0) ? total_size : 0, disp_unit,
                            MPI_INFO_NULL, MPI_COMM_WORLD, &base, &win);

    // all ranks get shared pointer from rank 0
    MPI_Aint size;
    int dispu;
    MPI_Win_shared_query(win, 0, &size, &dispu, &base);

    float *leftMat  = base;
    float *rightMat = base + m * n;
    float *resMat   = base + m * n + n * k;

    // init on rank 0
    if (rank == 0) {
        for (int i = 0; i < m * n; i++) leftMat[i] = 1.0;
        for (int i = 0; i < n * k; i++) rightMat[i] = 1.0;
        for (int i = 0; i < m * k; i++) resMat[i] = 0.0;
    }

    // passive-target epoch: lock + sync + barrier + sync => visibility
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    MPI_Win_sync(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(win);

    int rows_per_proc = m / worldSize;
    int my_rows = (rank == worldSize - 1) ? m - rank * rows_per_proc : rows_per_proc;
    int my_start = rank * rows_per_proc;

    // each process creates local transposed copy for cache-friendly access
    float *rightT = new float[k * n];
    for (int r = 0; r < n; r++)
        for (int c = 0; c < k; c++)
            rightT[c * n + r] = rightMat[r * k + c];

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    for (int row = 0; row < my_rows; row++) {
        int global_row = my_start + row;
        for (int col = 0; col < k; col++) {
            float sum = 0.0;
            for (int i = 0; i < n; i++)
                sum += leftMat[global_row * n + i] * rightT[col * n + i];
            resMat[global_row * k + col] = sum;
        }
    }
    delete[] rightT;

    // flush all writes, make them visible to rank 0
    MPI_Win_sync(win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(win);

    gettimeofday(&stop, NULL);

    MPI_Win_unlock_all(win);

    double t = (stop.tv_sec - start.tv_sec) * 1000.0 +
               (stop.tv_usec - start.tv_usec) / 1000.0;

    if (rank == 0) {
        cout << "mpi-shm matmul: " << t << " ms" << endl;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                if (int(resMat[i * k + j]) != n) {
                    cout << resMat[i * k + j] << " error\n";
                    exit(-1);
                }
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
