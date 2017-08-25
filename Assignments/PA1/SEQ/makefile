CC = gcc
EXEC = jacobi_1D_SEQ jacobi_2D_SEQ  mat_vec_SEQ
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 

all: $(EXEC)
jacobi_1D_SEQ: jacobi_1D.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ jacobi_1D.c  $(OBJS)


jacobi_2D_SEQ: jacobi_2D.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ jacobi_2D.c $(OBJS)


mat_vec_SEQ: mat_vec.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ mat_vec.c $(OBJS)


timer.o: timer.c
	$(CC) $(SEQFLAGS) -o $@ -c timer.c

clean: 
	rm -f *.o $(OBJS) $(EXEC)
