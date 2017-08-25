CC = gcc
EXEC = prog1_SEQ prog2_SEQ  prog1_OMP1 prog1_OMP2 #prog2_OMP
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 

all: $(EXEC)
prog1_SEQ: prog1.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ prog1.c  $(OBJS) -DSEQ

prog1_OMP1: prog1.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ prog1.c  $(OBJS) -DPAR1

prog1_OMP2: prog1.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ prog1.c  $(OBJS) -DPAR2

prog2_SEQ: prog2.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ prog2.c $(OBJS) -DSEQ


#prog2_OMP: prog2.c $(OBJS) $(H_FILE)
#	$(CC) $(FLAGS) $(MATHFLAG) -o $@ prog2.c $(OBJS) -DPAR


timer.o: timer.c
	$(CC) $(SEQFLAGS) -o $@ -c timer.c

clean: 
	rm -f *.o $(OBJS) $(EXEC)
