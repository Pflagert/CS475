CC = gcc
EXEC = mandSEQ mandOMP
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3

all: $(EXEC)

mandSEQ: mandelbrot.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) -o $@ mandelbrot.c  $(MATHFLAG) $(OBJS)

mandOMP: mandelbrot.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) -o $@ mandelbrot.c $(MATHFLAG) $(OBJS)


timer.o: timer.c
	$(CC) $(SEQFLAGS) -o $@ -c timer.c

clean: 
	rm -f *.o *.pgm $(OBJS) $(EXEC)
