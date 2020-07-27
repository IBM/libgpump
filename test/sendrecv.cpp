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
#include <libgpump_internal.h>
#include <assert.h>
#include <unistd.h>
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
    gpump g ;
    g.init(MPI_COMM_WORLD) ;
    int size ;
    MPI_Comm_size(MPI_COMM_WORLD, &size) ;
    int rank ;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
    cudaStream_t stream ;
    CUDA_CHECK(cudaStreamCreate(&stream)) ;
    char * send_buffer ;
    CUDA_CHECK(cudaMalloc(&send_buffer, k_BufferSize)) ;
    CUDA_CHECK(cudaMemset(send_buffer, rank, k_BufferSize)) ;
    ibv_mr * send_mr = g.register_region(send_buffer, k_BufferSize) ;
    char * recv_buffer ;
    CUDA_CHECK(cudaMalloc(&recv_buffer, k_BufferSize)) ;
    CUDA_CHECK(cudaMemset(recv_buffer, -1, k_BufferSize)) ;
    ibv_mr * recv_mr = g.register_region(recv_buffer, k_BufferSize) ;
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank ) g.connect_propose(i) ;
      }
    for ( int i=0; i<size; i+=1)
      {
        if ( i != rank ) g.connect_accept(i) ;
      }
//    for ( int i=0; i<size; i+=1)
//      {
//        if ( i != rank )
//          {
//            g.connect(i) ;
//          }
//      }
    g.receive(1-rank, recv_mr, 0, k_BufferSize) ;
    MPI_Barrier(MPI_COMM_WORLD); /* ensure receive is posted before the send */
    if( 0 == rank ) sleep (20);
    g.stream_send(1-rank, stream, send_mr, 0, k_BufferSize) ;
    g.stream_wait_send_complete(1-rank, stream) ;
    g.wait_send_complete(1-rank) ;
    g.stream_wait_recv_complete(1-rank, stream) ;
    printf("Posted wait for recv_cq\n");
    CUDA_CHECK(cudaStreamSynchronize(stream)) ;
    printf("Synchronized completed after wait\n");
    g.wait_receive_complete(1-rank) ;
    int sample ;
    CUDA_CHECK(cudaMemcpy(&sample, recv_buffer, sizeof(int),cudaMemcpyDeviceToHost)) ;
    fprintf(stderr, "Result of receive: 0x%08x on rank %d\n", sample, rank) ;
    CUDA_CHECK(cudaStreamDestroy(stream)) ;
    g.deregister_region(send_mr) ;
    g.deregister_region(recv_mr) ;
    g.term() ;
    MPI_Finalize() ;
    CUDA_CHECK(cudaFree(send_buffer)) ;
    CUDA_CHECK(cudaFree(recv_buffer)) ;
    return 0 ;
  }
