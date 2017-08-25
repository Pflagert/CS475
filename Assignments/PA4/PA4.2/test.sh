#!/bin/bash

#PBS -N MPI-VERT
#PBS -l walltime=01:00:00
cd /home/pflagert/lustrefs/
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib
aprun -n1 -d2 ./jacobi_1D 4000 200000