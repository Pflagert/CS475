CC = gcc
OBJS = timer.o
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 
MATHFLAGS = -lm
EXEC = sieve 

all: $(EXEC)

sieve: sieve.c $(OBJS)
	$(CC) $(SEQFLAGS)  -o $@ $< $(OBJS) $(MATHFLAGS)

timer.o: timer.c
	$(CC) -O3 -o $@ -c $<

clean:
	rm -f $(OBJS) $(EXEC)
