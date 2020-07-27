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
#include <infiniband/verbs.h>
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
  k_BufferSize = 8388608
};
int main(int argc, char **argv)
  {
    MPI_Init(&argc, &argv) ;
    struct gpump *g = gpump_init(MPI_COMM_WORLD);
    int size ;
    MPI_Comm_size(MPI_COMM_WORLD, &size) ;
    int rank ;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
    cudaStream_t stream ;
    CUDA_CHECK(cudaStreamCreate(&stream)) ;
    char * send_buffer ;
    CUDA_CHECK(cudaMalloc((void **)&send_buffer, k_BufferSize)) ;
    CUDA_CHECK(cudaMemset(send_buffer, rank, k_BufferSize)) ;
    struct ibv_mr * send_mr = gpump_register_region(g, send_buffer, k_BufferSize) ;
    char * recv_buffer ;
    CUDA_CHECK(cudaMalloc((void **)&recv_buffer, k_BufferSize)) ;
    CUDA_CHECK(cudaMemset(recv_buffer, -1, k_BufferSize)) ;
    struct ibv_mr * recv_mr = gpump_register_region(g, recv_buffer, k_BufferSize) ;
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank ) gpump_connect_propose(g,i) ;
      }
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank ) gpump_connect_accept(g,i) ;
      }
    gpump_receive(g, 1-rank, recv_mr, 0, k_BufferSize) ;
    MPI_Barrier(MPI_COMM_WORLD) ;
    gpump_stream_send(g, 1-rank, stream, send_mr, 0, k_BufferSize) ;
    gpump_stream_wait_send_complete(g, 1-rank, stream) ;
    gpump_wait_send_complete(g, 1-rank) ;
    gpump_stream_wait_recv_complete(g,1-rank, stream) ;
    gpump_wait_receive_complete(g, 1-rank) ;
    for ( int j=0; j<10000; j += 1 )
      {
        gpump_receive(g, 1-rank, recv_mr, 0, k_BufferSize) ;
        gpump_stream_send(g, 1-rank, stream, send_mr, 0, k_BufferSize) ;
        gpump_stream_wait_send_complete(g, 1-rank, stream) ;
        gpump_wait_send_complete(g, 1-rank) ;
        gpump_stream_wait_recv_complete(g,1-rank, stream) ;
        gpump_wait_receive_complete(g, 1-rank) ;
      }
    int sample ;
    CUDA_CHECK(cudaMemcpy(&sample, recv_buffer, sizeof(int),cudaMemcpyDeviceToHost)) ;
    fprintf(stderr, "Result of receive: 0x%08x on rank %d\n", sample, rank) ;
    CUDA_CHECK(cudaStreamDestroy(stream)) ;
    gpump_deregister_region(g, send_mr) ;
    gpump_deregister_region(g, recv_mr) ;
    gpump_term(g) ;
    MPI_Finalize() ;
    CUDA_CHECK(cudaFree(send_buffer)) ;
    CUDA_CHECK(cudaFree(recv_buffer)) ;
    return 0 ;
  }
