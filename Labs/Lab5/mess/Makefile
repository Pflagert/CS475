CC       = mpicc

EXES = cycle exchange sat

all: cycle exchange sat

cycle: cycle.c
	$(CC) -o cycle cycle.c -O3


exchange: exchange.c
	$(CC) -o exchange exchange.c -O3


sat: sat.c
	$(CC) -o sat sat.c -O3

clean:
	rm -f $(EXES)


