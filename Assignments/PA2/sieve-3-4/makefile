CC = gcc
OBJS = timer.o
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 
MATHFLAGS= -lm

SEQEXEC= sieve3 
EXEC= sieve4

all: $(SEQEXEC) $(EXEC)

$(SEQEXEC): sieve.c $(OBJS)
	$(CC) $(SEQFLAGS) -o $@ $< $(OBJS) $(MATHFLAGS)
$(EXEC): sieve.c $(OBJS)
	$(CC) $(FLAGS) -o $@ $< $(OBJS) $(MATHFLAGS)
timer.o: timer.c
	$(CC) -O3 -o $@ -c $<

clean:
	rm -f $(OBJS) $(SEQEXEC) $(EXEC)
