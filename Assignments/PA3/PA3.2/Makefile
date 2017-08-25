
CC = gcc
OBJS = timer.o
FLAGS = -fopenmp -O3
SEQFLAGS = -O3 
TAR = PA3.2.tar
EXEC = knapSeq knap1 knap2-V1 knap2-V1 knap2-V2 knap2-V3 knap3 knap4


all: $(EXEC)
knapSeq: knap.c timer.o
	$(CC) $(FLAGS) -o knapSeq knap.c $(OBJS)
knap1: knap1.c timer.o
	$(CC) $(FLAGS) -o knap1 knap1.c $(OBJS)
knap2-V1: knap2-V1.c timer.o
	$(CC) $(FLAGS) -o knap2-V1 knap2-V1.c $(OBJS)
knap2-V2: knap2-V2.c timer.o
	$(CC) $(FLAGS) -o knap2-V2 knap2-V2.c $(OBJS)
knap2-V3: knap2-V3.c timer.o
	$(CC) $(FLAGS) -o knap2-V3 knap2-V3.c $(OBJS)
knap3: knap3.c timer.o
	$(CC) $(FLAGS) -o knap3 knap3.c $(OBJS)
knap4: knap4.c timer.o
	$(CC) $(FLAGS) -o knap4 knap4.c $(OBJS)

timer.o: timer.c
	$(CC) -o $@ -c timer.c
package: clean
	tar -cvf $(TAR) timer.c timer.h knap.c knap1.c knap2.c knap3.c knap4.c Makefile
clean:
	rm -f $(EXEC) $(OBJS) $(TAR)
