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

enum {
  k_DEVICE_WINDOW_SIZE = 1084576
};
int main(int argc, char **argv)
  {
    MPI_Init(&argc, &argv) ;
    int size ;
    MPI_Comm_size(MPI_COMM_WORLD, &size) ;
    int rank ;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
    cudaStream_t stream ;
    CUDA_CHECK(cudaStreamCreate(&stream)) ;
    char * device_local_window ;
    CUDA_CHECK(cudaMalloc((void **)&device_local_window, k_DEVICE_WINDOW_SIZE));
//    cout << "device_local_window=" << (void *) device_local_window << endl ;
    char * device_remote_window ;
    CUDA_CHECK(cudaMalloc((void **)&device_remote_window, k_DEVICE_WINDOW_SIZE));
//    cout << "device_remote_window=" << (void *) device_remote_window << endl ;
    CUDA_CHECK(cudaMemset(device_local_window, rank, k_DEVICE_WINDOW_SIZE)) ;
    CUDA_CHECK(cudaMemset(device_remote_window, -1, k_DEVICE_WINDOW_SIZE)) ;
    struct gpump *g =gpump_init(MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize() ) ;
    MPI_Barrier(MPI_COMM_WORLD) ;
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank ) gpump_connect_propose(g,i) ;
      }
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank ) gpump_connect_accept(g,i) ;
      }
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank ) gpump_create_window_propose(g, i, device_local_window,device_remote_window, k_DEVICE_WINDOW_SIZE) ;
      }
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank ) gpump_window_accept(g, i) ;
      }
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank )
          {
            gpump_stream_put(g, i, stream,0, 0,k_DEVICE_WINDOW_SIZE ) ;
            gpump_stream_wait_put_complete(g, i, stream) ;
            CUDA_CHECK(cudaStreamSynchronize(stream)) ;
            gpump_wait_put_complete(g, i) ;
            MPI_Barrier(MPI_COMM_WORLD) ;
            int sample ;
            CUDA_CHECK(cudaMemcpy(&sample, device_remote_window, sizeof(int),cudaMemcpyDeviceToHost)) ;
            fprintf(stderr, "Result of remote put: 0x%08x on rank %d\n", sample, rank) ;
          }
      }
    CUDA_CHECK(cudaMemset(device_local_window, -1, k_DEVICE_WINDOW_SIZE)) ;
    CUDA_CHECK(cudaMemset(device_remote_window, rank, k_DEVICE_WINDOW_SIZE)) ;
    CUDA_CHECK(cudaDeviceSynchronize() ) ;
    MPI_Barrier(MPI_COMM_WORLD) ;
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank )
          {
            gpump_stream_get(g, i, stream,0, 0,k_DEVICE_WINDOW_SIZE) ;
            gpump_stream_wait_get_complete(g, i, stream) ;
            CUDA_CHECK(cudaStreamSynchronize(stream)) ;
            gpump_wait_get_complete(g, i) ;
//            MPI_Barrier(MPI_COMM_WORLD) ;
            int sample ;
            CUDA_CHECK(cudaMemcpy(&sample, device_local_window, sizeof(int),cudaMemcpyDeviceToHost)) ;
            fprintf(stderr, "Result of remote get: 0x%08x on rank %d\n", sample, rank) ;
         }
      }
    MPI_Barrier(MPI_COMM_WORLD) ;
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank )
          {
            gpump_destroy_window(g, i) ;
          }
      }
    CUDA_CHECK(cudaStreamDestroy(stream)) ;
    gpump_term(g) ;
    MPI_Finalize() ;
    return 0 ;
  }
