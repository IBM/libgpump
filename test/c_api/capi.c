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
#include <infiniband/verbs.h>
#include <cuda.h>
#include <cuda_runtime.h>
int main(int argc, char **argv)
  {
    MPI_Init(&argc, &argv) ;
    struct gpump *g = gpump_init(MPI_COMM_WORLD) ;
    gpump_term(g) ;
    char buffer[30000] ;
    char remote_buffer[30000] ;
    struct ibv_mr *mrp = gpump_register_region(g,buffer, sizeof(buffer)) ;
    struct ibv_mr *rmrp = gpump_register_region(g,remote_buffer, sizeof(remote_buffer)) ;
    gpump_connect_propose(g, 0) ;
    gpump_connect_accept(g, 0) ;
    gpump_create_window_propose(g, 0, buffer, remote_buffer, sizeof(buffer) ) ;
    gpump_window_accept(g, 0) ;
    gpump_replace_window_propose(g, 0, buffer, remote_buffer, sizeof(buffer) ) ;
    gpump_window_accept(g, 0) ;
    int wx ;
    gpump_create_window_propose_x(g, 0, &wx, buffer, remote_buffer, sizeof(buffer) ) ;
    gpump_window_accept_x(g, 0, wx ) ;
    gpump_replace_window_propose_x(g, 0, wx, buffer, remote_buffer, sizeof(buffer) ) ;
    gpump_window_accept_x(g, 0, wx ) ;
    gpump_cork(g) ;
    cudaStream_t stream ;
    gpump_uncork(g, stream) ;

    gpump_stream_put(g, 0, stream, 0, 0, sizeof(buffer) ) ;
    gpump_iput(g, 0, 0, 0, sizeof(buffer) ) ;
    gpump_stream_wait_put_complete(g, 0,  stream ) ;
    gpump_cpu_ack_iput(g, 0) ;
    int is_complete=gpump_is_put_complete(g, 0) ;
    gpump_wait_put_complete(g, 0) ;
    gpump_stream_get(g, 0, stream, 0, 0, sizeof(buffer) ) ;
    gpump_iget(g, 0, 0, 0, sizeof(buffer) ) ;
    gpump_stream_wait_get_complete(g, 0,  stream ) ;
    gpump_cpu_ack_iget(g, 0) ;
    is_complete=gpump_is_get_complete(g, 0) ;
    gpump_wait_get_complete(g, 0) ;

    gpump_stream_put_x(g, 0, wx, stream, 0, 0, sizeof(buffer) ) ;
    gpump_iput_x(g, 0, wx, 0, 0, sizeof(buffer) ) ;
    gpump_stream_wait_put_complete_x(g, 0, wx,  stream ) ;
    gpump_cpu_ack_iput_x(g, 0, wx) ;
    is_complete=gpump_is_put_complete_x(g, 0, wx) ;
    gpump_wait_put_complete_x(g, 0, wx) ;
    gpump_stream_get_x(g, 0, wx, stream, 0, 0, sizeof(buffer) ) ;
    gpump_iget_x(g, 0, wx, 0, 0, sizeof(buffer) ) ;
    gpump_stream_wait_get_complete_x(g, 0, wx,  stream ) ;
    gpump_cpu_ack_iget_x(g, 0, wx) ;
    is_complete=gpump_is_get_complete_x(g, 0, wx) ;
    gpump_wait_get_complete_x(g, 0, wx) ;

    gpump_stream_send(g, 0, stream, mrp, 0, sizeof(buffer) ) ;
    gpump_isend(g, 0, mrp, 0, sizeof(buffer) ) ;
    gpump_stream_wait_send_complete(g, 0, stream) ;
    gpump_cpu_ack_isend(g, 0) ;
    is_complete=gpump_is_send_complete(g, 0) ;

    gpump_wait_send_complete(g, 0) ;
    gpump_receive(g,0, rmrp, 0, sizeof(remote_buffer) ) ;
    gpump_stream_wait_recv_complete(g, 0, stream) ;
    gpump_cpu_ack_recv(g, 0) ;
    is_complete=gpump_is_receive_complete(g, 0) ;
    gpump_wait_receive_complete(g, 0) ;

    gpump_destroy_window_x(g, 0, wx) ;
    gpump_destroy_window(g, 0) ;
    gpump_disconnect(g, 0) ;
    gpump_deregister_region(g,mrp) ;
    gpump_deregister_region(g,rmrp) ;
    MPI_Finalize() ;
    return 0 ;
  }
