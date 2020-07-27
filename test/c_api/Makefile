# Copyright (C) IBM Corporation 2018. All Rights Reserved
#
#    This program is licensed under the terms of the Eclipse Public License
#    v1.0 as published by the Eclipse Foundation and available at
#    http://www.eclipse.org/legal/epl-v10.html
#
#    
#    
# $COPYRIGHT$
MPI_ROOT ?= /opt/ibm/spectrum_mpi

all: capi.o capi

capi.o: capi.c
	mpicc -g -c -I ${MPI_ROOT}/libgpump/include  -I /usr/local/cuda/include capi.c
	
capi: capi.o
	mpicc -o capi -L ${MPI_ROOT}/libgpump/lib capi.o -lgpump

clean:
	rm -f capi.o capi
