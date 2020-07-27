#!/bin/bash -x
# Copyright (C) IBM Corporation 2018. All Rights Reserved
#
#    This program is licensed under the terms of the Eclipse Public License
#    v1.0 as published by the Eclipse Foundation and available at
#    http://www.eclipse.org/legal/epl-v10.html
#
#    
#    
# $COPYRIGHT$
export LD_LIBRARY_PATH=${MPI_ROOT}/libgpump/lib:${LD_LIBRARY_PATH}
mpirun -np 2 -hostfile hostfile \
       -x LD_LIBRARY_PATH \
       -x  GDS_ENABLE_DEBUG=1 \
       sendrecv_nostream
