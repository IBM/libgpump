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
#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#define CUDA_CHECK(stmt)                                \
do {                                                    \
    cudaError_t result = (stmt);                        \
    if (cudaSuccess != result) {                        \
        fprintf(stderr, "[%s:%d] cuda failed with %s \n",   \
         __FILE__, __LINE__,cudaGetErrorString(result));\
        exit(-1);                                       \
    }                                                   \
    assert(cudaSuccess == result);                      \
} while (0)

int main(int argc, char **argv)
  {
    MPI_Init(&argc, &argv) ;
    int size ;
    MPI_Comm_size(MPI_COMM_WORLD, &size) ;
    int rank ;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
    cudaStream_t stream ;
    CUDA_CHECK(cudaStreamCreate(&stream)) ;
    struct gpump *g =gpump_init(MPI_COMM_WORLD);
    for ( int rep=0 ; rep<100; rep+=1)
      {
        for ( int i=0; i<size; i+=1)
          {
            if ( i != rank )
              {
                gpump_connect_propose(g,i) ;
                gpump_connect_accept(g,i) ;
                gpump_disconnect(g,i) ;
              }
          }
      }
    CUDA_CHECK(cudaStreamDestroy(stream)) ;
    gpump_term(g) ;
    MPI_Finalize() ;
    return 0 ;
  }
