CC = mpic++
CCFLAGS = -O2 -fopenmp -I$(HOME)/openblas-install/include
LDFLAGS = -L$(HOME)/openblas-install/lib -lopenblas -lpthread -lgfortran

all: gemm conv pooling

gemm: gemm.cpp
	${CC} ${CCFLAGS} gemm.cpp -o gemm ${LDFLAGS}

conv: conv.cpp
	${CC} ${CCFLAGS} conv.cpp -o conv ${LDFLAGS}

pooling: pooling.cpp
	${CC} ${CCFLAGS} pooling.cpp -o pooling

clean:
	rm -f gemm conv pooling
