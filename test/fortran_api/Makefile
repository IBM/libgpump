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

all: forapi.o forapi

forapi.o: forapi.F90
	mpifort -g -c -I ${MPI_ROOT}/libgpump/mod forapi.F90
	
forapi: forapi.o
	mpifort -o forapi -L ${MPI_ROOT}/libgpump/lib forapi.o -lgpump

clean:
	rm -f forapi.o forapi
	