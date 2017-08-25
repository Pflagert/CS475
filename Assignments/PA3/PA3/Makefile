
CC = gcc
OBJS = timer.o
FLAGS = -fopenmp -O3
SEQFLAGS = -O3 
TAR = PA3.2.tar
EXEC = knap1 knap2 knap3


all: $(EXEC)

knap1: knap1.c timer.o
	$(CC) $(FLAGS) -o knap1 knap1.c $(OBJS)
knap2: knap2.c timer.o
	$(CC) $(FLAGS) -o knap2 knap2.c $(OBJS)
knap3: knap3.c timer.o
	$(CC) $(FLAGS) -o knap3 knap3.c $(OBJS)

timer.o: timer.c
	$(CC) -o $@ -c timer.c
package: clean
	tar -cvf $(TAR) knap1.c knap2.c knap3.c Makefile
clean:
	rm -f $(EXEC) $(OBJS) $(TAR)
