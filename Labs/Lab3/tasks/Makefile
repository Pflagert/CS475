CC = gcc
OBJS = timer.o
FLAGS = -fopenmp -O3
DEBUGFLAG = -DDEBUG
SEQFLAGS = -O3 -lm

EXEC = Merge_SEQ Merge_SEQ_debug Merge Merge_debug

all: $(EXEC)


Merge_SEQ: Merge_sort.c $(OBJS)
	$(CC) $(SEQFLAGS)  -o $@ $< $(OBJS)
Merge: Merge_sort.c $(OBJS)
	$(CC) $(FLAGS)  -o $@ $< $(OBJS)
Merge_SEQ_debug:Merge_sort.c $(OBJS)
	$(CC) $(SEQFLAGS) $(DEBUGFLAG)  -o $@ $< $(OBJS)
Merge_debug:Merge_sort.c $(OBJS)
	$(CC) $(FLAGS) $(DEBUGFLAG)  -o $@ $< $(OBJS)
timer.o: timer.c
	$(CC) -O3 -o $@ -c $<

clean:
	rm -f $(OBJS) $(EXEC)
