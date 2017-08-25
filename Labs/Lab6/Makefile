CC = nvcc

EXE = hello

all: $(EXE)

clean:
	rm -f $(EXE)

hello: HelloWorld.cu
	$(CC) HelloWorld.cu -o $@