CC = g++
MPICC = mpic++

all: serial openmp mpi-shm

serial: serial.cpp
	$(CC) -O2 -o serial serial.cpp

openmp: openmp.cpp
	$(CC) -O2 -fopenmp -o openmp openmp.cpp

mpi-shm: mpi-shm.cpp
	$(MPICC) -O2 -o mpi-shm mpi-shm.cpp

clean:
	rm -f serial openmp mpi-shm
