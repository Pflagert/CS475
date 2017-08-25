#CC=mpicc #On CS machines
CC=cc #On Cray
TAR=PA4.2.tar

EXES = Jacobi2D-BlockParallel-MPI Jacobi2D-BlockParallel-MPI-VERT

all: Jacobi2D-BlockParallel-MPI Jacobi2D-BlockParallel-MPI-VERT

Jacobi2D-BlockParallel-MPI:  Jacobi2D-BlockParallel-MPI.test.c util.h
	$(CC) -o Jacobi2D-BlockParallel-MPI Jacobi2D-BlockParallel-MPI.test.c -O3

Jacobi2D-BlockParallel-MPI-VERT: Jacobi2D-BlockParallel-MPI-VERT.test.c util.h
	$(CC) -o Jacobi2D-BlockParallel-MPI-VERT Jacobi2D-BlockParallel-MPI-VERT.test.c -O3


clean:
	rm -f $(EXES) $(TAR)

package: clean
	tar -cvf $(TAR) Jacobi2D-BlockParallel-MPI-VERT.test.c Jacobi2D-BlockParallel-MPI.test.c util.h Makefile

send: package
	scp $(TAR) pflagert@lincoln.cs.colostate.edu:~/CS/cs475/Assignments/PA4/
