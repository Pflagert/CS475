VPATH=matmult:func
SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
TAR=PA6.tar
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 #-std=c99 -lm
#--maxrregcount=100 --ptxas-options -v 

CHECK_FOR_NVCC_COMP := $(shell nvcc --version 2>/dev/null)
ifdef CHECK_FOR_NVCC_COMP
	NVCC=nvcc
	OPTIONS   := -G -g -O3 -D_FORCE_INLINES
else
	NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
	OPTIONS   :=  -O3
endif

EXECS :=  bpl_CPU bpl_GPU
all:$(EXECS)

clean:
	rm -f $(EXECS) *.o $(TAR)

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#Example make commands.
#Modify to include the Kernels that you create
matmultKernels.o : matmultKernels.cu matmultKernels.h
	${NVCC} $< -c -o $@ $(OPTIONS)
functionKernels.o : functionKernels.cu functionKernels.h
	${NVCC} $< -c -o $@ $(OPTIONS)

bpl_GPU : bpl.cu matmultKernels.h matmultKernels.o functionKernels.h functionKernels.o timer.o
	${NVCC} $< matmultKernels.o functionKernels.o -o $@ $(LIB) timer.o $(OPTIONS)

bpl_CPU : bpl_cpu.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)
  

remake: clean all

package: clean
	tar -cvf $(TAR) *
send: package
	scp $(TAR) pflagert@lincoln.cs.colostate.edu:~/CS/cs475/Assignments/