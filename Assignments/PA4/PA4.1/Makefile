CC=mpicc #On CS machines
#CC=cc #On Cray
TAR=PA4.1.tar

EXES = Jacobi2D-BlockParallel-MPI Jacobi2D-BlockParallel-MPI-VERT

all: Jacobi2D-BlockParallel-MPI Jacobi2D-BlockParallel-MPI-VERT

Jacobi2D-BlockParallel-MPI:  Jacobi2D-BlockParallel-MPI.test.c
	$(CC) -o Jacobi2D-BlockParallel-MPI Jacobi2D-BlockParallel-MPI.test.c -O3

Jacobi2D-BlockParallel-MPI-VERT: Jacobi2D-BlockParallel-MPI-VERT.test.c
	$(CC) -o Jacobi2D-BlockParallel-MPI-VERT Jacobi2D-BlockParallel-MPI-VERT.test.c -O3


clean:
	rm -f $(EXES) $(TAR)

package: clean
	tar -cvf $(TAR) Makefile util.h Jacobi2D-BlockParallel-MPI.test.c Jacobi2D-BlockParallel-MPI-VERT.test.c
