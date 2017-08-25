CC=gcc
CHPL=chpl
MPICC=mpicc
BIN=bin
SRC=src

#SDK_INSTALL_PATH :=  /usr/local/cuda-6.5/bin/
NVCC=$(SDK_INSTALL_PATH)nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 --maxrregcount=100 --ptxas-options -v
# CUDA code generation flags

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30)
  
OPTIONS   :=  -O3 -ftz=false $(GENCODE_FLAGS) -DCOUNT


BASELINES = Jacobi1D-Serial-C\
						Jacobi2D-Serial-C

MPIVARIANTS = Jacobi2D-BlockParallel-MPI

CVARIANTS = Jacobi1D-DiamondByHand-OMP_dyn\
						Jacobi1D-DiamondByHand-OMP_static\
						Jacobi1D-DiamondSlabByHand-OMP_dyn\
						Jacobi1D-NaiveParallel-OMP_dyn\
						Jacobi1D-NaiveParallel-OMP_static\
						Jacobi2D-DiamondByHandParam-OMP_dyn\
						Jacobi2D-DiamondByHandParam-OMP_static\
						Jacobi2D-DiamondStatic276-OMP\
						Jacobi2D-NaiveParallel-OMP_dyn\
						Jacobi2D-NaiveParallel-OMP_static\
						Jacobi2D-NaiveParallelSpaceTiled-OMP\
						Jacobi2D-DiamondSlabByHandParam-OMP\
						Seismic-NaiveParallel-OMP
						#Jacobi2D-DiamondByHandParamOpt-OMP\
						#Jacobi2D-DiamondISCCParam-OMP\
						Jacobi2D-DiamondSlabISCCParam-OMP\
						Jacobi2D-DiamondSlabISCCSpecializedBounds-OMP\

CHPLVARIANTS = Jacobi1D-DiamondByHand-Chapel_dyn\
               Jacobi1D-DiamondByHand-Chapel_static\
               Jacobi1D-DiamondSlabByHand-Chapel\
               Jacobi1D-NaiveParallel-Chapel_dyn\
               Jacobi1D-NaiveParallel-Chapel_static\
               Jacobi2D-DiamondByHandParam-Chapel_dyn\
               Jacobi2D-DiamondByHandParam-Chapel_static\
               Jacobi2D-NaiveParallel-Chapel_dyn\
               Jacobi2D-NaiveParallel-Chapel_static\
               Seismic-NaiveParallel-Chapel

CUDAVARIANTS = Jacobi1D-NaiveParallelGlobal-CUDA\
		Jacobi1D-NaiveParallelGlobalLikeShared-CUDA \
		Jacobi1D-NaiveParallelShared-CUDA\
		Jacobi1D-NaiveParallelSharedApproxPerfModelIn-CUDA\
		Jacobi2D-NaiveParallelSharedApproxPerfModelIn-CUDA\
		Jacobi1D-NaiveParallelSharedApproxPerfModelInTempBuffer-CUDA\
		Jacobi1D-NaiveParallelSharedPingPong-CUDA\
		Jacobi1D-DiamondByHand-CUDA\
		Jacobi2D-NaiveParallelGlobal-CUDA\
		Jacobi2D-NaiveParallelShared-CUDA\
		Jacobi2D-DiamondSlabByHandHACKED-CUDA\
		Jacobi2D-DiamondSlabByHandParam-CUDA\
		Jacobi1D-NeighSyncParallelGlobal-CUDA\
		Jacobi1D-DiamondSlabByHand-CUDA



all: $(BASEINES) $(CVARIANTS) $(CHPLVARIANTS) 
	#$(CUDAVARIANTS)

baselines: $(BASELINES)

mpi: $(CVARIANTS)

omp: $(CVARIANTS)

chapel: $(CHPLVARIANTS)

cuda: $(CUDAVARIANTS)

$(BASELINES): % : $(SRC)/%.test.c
	$(CC) `cat $(SRC)/$@.perfcompopts` $(SRC)/$@.test.c -o $(BIN)/$@

$(MPIVARIANTS): % : $(SRC)/%.test.c
	$(MPICC) `cat $(SRC)/$@.perfcompopts` $(SRC)/$@.test.c -o $(BIN)/$@

$(CVARIANTS): % : $(SRC)/%.test.c
	$(CC) `cat $(SRC)/$@.perfcompopts` $(SRC)/$@.test.c -o $(BIN)/$@

$(CHPLVARIANTS): % : $(SRC)/%.chpl
	$(CHPL) `cat $(SRC)/$@.perfcompopts` $(SRC)/$@.chpl -o $(BIN)/$@

$(CUDAVARIANTS): % : $(SRC)/%.test.cu
	$(NVCC) -g  $<  -o $(BIN)/$@ $(OPTIONS) -DCUDA

static:
	$(CC) -fopenmp -O4 -std=c99 -DTAU=132 $(SRC)/Jacobi2D-DiamondISCCstatic-OMP.test.c -o $(BIN)/Jacobi2D-DiamondISCCstatic-OMP-132
	$(CC) -fopenmp -O4 -std=c99 -DTAU=136 $(SRC)/Jacobi2D-DiamondISCCstatic-OMP.test.c -o $(BIN)/Jacobi2D-DiamondISCCstatic-OMP-136
	$(CC) -fopenmp -O4 -std=c99 -DTAU=166 $(SRC)/Jacobi2D-DiamondISCCstatic-OMP.test.c -o $(BIN)/Jacobi2D-DiamondISCCstatic-OMP-166
	$(CC) -fopenmp -O4 -std=c99 -DTAU=180 $(SRC)/Jacobi2D-DiamondISCCstatic-OMP.test.c -o $(BIN)/Jacobi2D-DiamondISCCstatic-OMP-180
	$(CC) -fopenmp -O4 -std=c99 -DTAU=244 $(SRC)/Jacobi2D-DiamondISCCstatic-OMP.test.c -o $(BIN)/Jacobi2D-DiamondISCCstatic-OMP-244
	$(CC) -fopenmp -O4 -std=c99 -DTAU=276 $(SRC)/Jacobi2D-DiamondISCCstatic-OMP.test.c -o $(BIN)/Jacobi2D-DiamondISCCstatic-OMP-276
	$(CC) -fopenmp -O4 -std=c99 -DTAU=344 $(SRC)/Jacobi2D-DiamondISCCstatic-OMP.test.c -o $(BIN)/Jacobi2D-DiamondISCCstatic-OMP-344

clean:
	rm bin/J*
