# Copyright (C) IBM Corporation 2018. All Rights Reserved
#
#    This program is licensed under the terms of the Eclipse Public License
#    v1.0 as published by the Eclipse Foundation and available at
#    http://www.eclipse.org/legal/epl-v10.html
#
#    
#    
# $COPYRIGHT$
MPI_ROOT?=/opt/ibm/spectrum_mpi
CXX=$(MPI_ROOT)/bin/mpicxx
CC=$(MPI_ROOT)/bin/mpicxx
CXXFLAGS=-g -D__STDC_FORMAT_MACROS -I ../include -I ../src -I /usr/local/cuda/include
LOADLIBES=-lgpump -libverbs -lcuda -lcudart
LDFLAGS=-L $(MPI_ROOT)/libgpump/lib -L /usr/local/cuda/lib64

all: init putget sendrecv putget_nostream sendrecv_nostream

putget: putget.o

putget_nostream: putget_nostream.o

sendrecv: sendrecv.o

sendrecv_nostream: sendrecv_nostream.o

clean:
	rm -f init init.o putget putget.o putget_nostream putget_nostream.o sendrecv sendrecv.o sendrecv_nostream sendrecv_nostream.o

init: init.o
	$(CXX) -L ../src $< -lgpump $(LDLIBS) -o $@
