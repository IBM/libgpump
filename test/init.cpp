// Copyright (C) IBM Corporation 2018. All Rights Reserved
//
//    This program is licensed under the terms of the Eclipse Public License
//    v1.0 as published by the Eclipse Foundation and available at
//    http://www.eclipse.org/legal/epl-v10.html
//
//    
//    
// $COPYRIGHT$

#include <mpi.h>
#include <libgpump.h>
#include <stdio.h>

int main(int argc, char **argv)
  {
    MPI_Init(&argc, &argv) ;
    printf("Initializing gpump library\n");
    gpump *g;
    g = gpump_init(MPI_COMM_WORLD) ;
    printf("Finalizing gpump library\n");
    gpump_term(g) ;
    MPI_Finalize() ;
    return 0 ;
  }
