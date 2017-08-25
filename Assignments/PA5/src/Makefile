###
### CS575 and CS575dl Spring 2011
### Wim Bohm, instructor
### Makefile for CUDA1 assignment
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
### Created: 2011-01-27 DVN
### Last Modified: Nov 2014 WB, WR, LR
###

# This tells make where to look for source files
VPATH=timer:matmult:vecadd
SDK_INSTALL_PATH :=  /usr/local/cuda
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
#--maxrregcount=100 --ptxas-options -v 
TAR  := PA5.2.tar
EXECS :=  vecadd00 vecadd01 matmult00 matmult01

CHECK_FOR_NVCC_COMP := $(shell nvcc --version 2>/dev/null)
ifdef CHECK_FOR_NVCC_COMP
	NVCC=nvcc
	OPTIONS   :=  -O3 -D_FORCE_INLINES
else
	NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
	OPTIONS   :=  -O3
endif


# Hide output from compile commands.  If you want to see the out put run `make VERBOSE=1`
ifndef VERBOSE
.SILENT:
endif

all: $(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) $(TAR) *.o

#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
matmultKernel00.o : matmultKernel00.cu matmultKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

matmult00 : matmult.cu  matmultKernel.h matmultKernel00.o timer.o
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32

#######################################################################
package: clean
	tar -cf $(TAR) *

send: package
	scp $(TAR) pflagert@lincoln.cs.colostate.edu:~/CS/cs475/Assignments/PA5/

