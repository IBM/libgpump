# Copyright (C) IBM Corporation 2018. All Rights Reserved
#
#    This program is licensed under the terms of the Eclipse Public License
#    v1.0 as published by the Eclipse Foundation and available at
#    http://www.eclipse.org/legal/epl-v10.html
#
#    
#    
# $COPYRIGHT$
CXX=$(MPI_ROOT)/bin/mpicxx

INCDIR?=/usr/local/cuda/include
ifeq ($(origin EXPORTDIR),undefined)
   $(error EXPORTDIR not set in environment)
endif

CXXFLAGS=-g -O3 -fPIC -D__STDC_FORMAT_MACROS -I ../include -I . -I ${INCDIR} -std=c++11

OFILES=gpump.o gpump_c.o gpump_f.o gdsync.o gdsync_apis.o gdsync_mem.o gdsync_memmgr.o gdsync_objs.o gpump_cuda_wrapper.o

all: $(EXPORTDIR)/libgpump/lib/libgpump.so.1 $(EXPORTDIR)/libgpump/lib/libgpump.so

#libgpump.a: $(OFILES)
#	$(AR) $(ARFLAGS) libgpump.a $(OFILES)

$(EXPORTDIR)/libgpump/lib/libgpump.so.1: $(OFILES)
	$(CXX) -shared -o $(EXPORTDIR)/libgpump/lib/libgpump.so.1 $(OFILES) -libverbs -ldl -Wl,-soname=libgpump.so.1

$(EXPORTDIR)/libgpump/lib/libgpump.so: $(EXPORTDIR)/libgpump/lib/libgpump.so.1
	ln -s libgpump.so.1 $(EXPORTDIR)/libgpump/lib/libgpump.so

clean:
	rm -f $(EXPORTDIR)/libgpump/lib/libgpump.so.1 $(EXPORTDIR)/libgpump/lib/libgpump.so
	rm -f $(OFILES) *.a
		
