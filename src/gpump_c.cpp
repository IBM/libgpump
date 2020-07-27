// Copyright (C) IBM Corporation 2018. All Rights Reserved
//
//    This program is licensed under the terms of the Eclipse Public License
//    v1.0 as published by the Eclipse Foundation and available at
//    http://www.eclipse.org/legal/epl-v10.html
//
//    
//    
// $COPYRIGHT$

#include <libgpump_internal.h>
#include <libgpump.h>
#include <stddef.h>
#include <stdlib.h>

struct gpump * gpump_init(MPI_Comm comm)
  {
    int rc;
    struct gpump * g = new struct gpump ;
    rc = g->init(comm) ;
    if(rc) {
      abort();
    }
    return g ;
  }
struct ibv_mr * gpump_register_region(struct gpump * g, void * addr, size_t size)
  {
    struct ibv_mr *mr = g->register_region(addr, size) ;
    return mr ;
  }
void gpump_deregister_region(struct gpump * g, struct ibv_mr * mr)
  {
    g->deregister_region(mr) ;
  }
void gpump_connect_propose(struct gpump * g, int target)
  {
    g->connect_propose(target) ;
  }
void gpump_connect_accept(struct gpump * g, int target)
  {
    g->connect_accept(target) ;
  }
//void gpump_connect(struct gpump * g, int target)
//  {
//    g->connect(target) ;
//  }
void gpump_disconnect(struct gpump * g, int target)
  {
    g->disconnect(target) ;
  }
void gpump_create_window_propose(struct gpump * g , int target, void * local_address, void * remote_address, size_t size)
  {
    g->create_window_propose(target, local_address, remote_address, size) ;
  }
void gpump_replace_window_propose(struct gpump * g , int target, void * local_address, void * remote_address, size_t size)
  {
    g->replace_window_propose(target, local_address, remote_address, size) ;
  }
void gpump_window_accept(struct gpump * g , int target)
  {
    g->window_accept(target) ;
  }
void gpump_create_window_propose_x(struct gpump * g , int target, int *wx, void * local_address, void * remote_address, size_t size)
  {
    g->create_window_propose_x(target, wx, local_address, remote_address, size) ;
  }
void gpump_replace_window_propose_x(struct gpump * g , int target, int wx, void * local_address, void * remote_address, size_t size)
  {
    g->replace_window_propose_x(target, wx, local_address, remote_address, size) ;
  }
void gpump_window_accept_x(struct gpump * g , int target, int wx)
  {
    g->window_accept_x(target, wx) ;
  }
void gpump_create_window(struct gpump * g , int target, void * local_address, void * remote_address, size_t size)
  {
    g->create_window(target, local_address, remote_address, size) ;
  }
void gpump_destroy_window(struct gpump * g , int target)
  {
    g->destroy_window(target) ;
  }
void gpump_destroy_window_x(struct gpump * g , int target, int wx)
  {
    g->destroy_window_x(target, wx) ;
  }
void gpump_cork(struct gpump * g)
  {
    g->cork() ;
  }
void gpump_uncork(struct gpump * g, cudaStream_t stream)
  {
    g->uncork(stream) ;
  }
void gpump_stream_put(struct gpump * g, int target, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size )
  {
    g->stream_put(target, stream,offset, remote_offset, size) ;
  }
void gpump_iput(struct gpump * g, int target, size_t offset, size_t remote_offset, size_t size )
  {
    g->iput(target, offset, remote_offset, size) ;
  }
void gpump_stream_wait_put_complete(struct gpump * g, int target, cudaStream_t stream)
  {
    g->stream_wait_put_complete(target, stream) ;
  }
void gpump_cpu_ack_iput(struct gpump *g, int target)
  {
    g->cpu_ack_iput(target) ;
  }
int gpump_is_put_complete(struct gpump * g, int target)
  {
    return g->is_put_complete(target) ;
  }
void gpump_wait_put_complete(struct gpump * g, int target)
  {
    g->wait_put_complete(target) ;
  }
void gpump_stream_get(struct gpump * g, int target, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size )
  {
    g->stream_get(target, stream, offset, remote_offset, size) ;
  }
void gpump_iget(struct gpump * g, int target, size_t offset, size_t remote_offset, size_t size )
  {
    g->iget(target, offset, remote_offset, size) ;
  }
void gpump_stream_wait_get_complete(struct gpump * g, int target, cudaStream_t stream)
  {
    g->stream_wait_get_complete(target, stream) ;
  }
void gpump_cpu_ack_iget(struct gpump *g, int target)
  {
    g->cpu_ack_iget(target) ;
  }
int gpump_is_get_complete(struct gpump * g, int target)
  {
    return g->is_get_complete(target) ;
  }
void gpump_wait_get_complete(struct gpump * g, int target)
  {
    g->wait_get_complete(target) ;
  }
void gpump_stream_put_x(struct gpump * g, int target, int wx, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size )
  {
    g->stream_put_x(target, wx, stream,offset, remote_offset, size) ;
  }
void gpump_iput_x(struct gpump * g, int target, int wx, size_t offset, size_t remote_offset, size_t size )
  {
    g->iput_x(target, wx, offset, remote_offset, size) ;
  }
void gpump_stream_wait_put_complete_x(struct gpump * g, int target, int wx, cudaStream_t stream)
  {
    g->stream_wait_put_complete_x(target, wx, stream) ;
  }
void gpump_cpu_ack_iput_x(struct gpump *g, int target, int wx)
  {
    g->cpu_ack_iput_x(target, wx) ;
  }
int gpump_is_put_complete_x(struct gpump * g, int target, int wx)
  {
    return g->is_put_complete_x(target, wx) ;
  }
void gpump_wait_put_complete_x(struct gpump * g, int target, int wx)
  {
    g->wait_put_complete_x(target, wx) ;
  }
void gpump_stream_get_x(struct gpump * g, int target, int wx, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size )
  {
    g->stream_get_x(target, wx, stream, offset, remote_offset, size) ;
  }
void gpump_iget_x(struct gpump * g, int target, int wx, size_t offset, size_t remote_offset, size_t size )
  {
    g->iget_x(target, wx, offset, remote_offset, size) ;
  }
void gpump_stream_wait_get_complete_x(struct gpump * g, int target, int wx, cudaStream_t stream)
  {
    g->stream_wait_get_complete_x(target, wx, stream) ;
  }
void gpump_cpu_ack_iget_x(struct gpump *g, int target, int wx)
  {
    g->cpu_ack_iget_x(target, wx) ;
  }
int gpump_is_get_complete_x(struct gpump * g, int target, int wx)
  {
    return g->is_get_complete_x(target, wx) ;
  }
void gpump_wait_get_complete_x(struct gpump * g, int target, int wx)
  {
    g->wait_get_complete_x(target, wx) ;
  }
void gpump_stream_send(struct gpump * g, int target, cudaStream_t stream, struct ibv_mr * source_mr, size_t offset, size_t size)
  {
    g->stream_send(target, stream, source_mr, offset, size) ;
  }
void gpump_isend(struct gpump * g, int target, struct ibv_mr * source_mr, size_t offset, size_t size)
  {
    g->isend(target, source_mr, offset, size) ;
  }
void gpump_stream_wait_send_complete(struct gpump * g, int target, cudaStream_t stream)
  {
    g->stream_wait_send_complete(target, stream) ;
  }
void gpump_cpu_ack_isend(struct gpump *g, int target)
  {
    g->cpu_ack_isend(target) ;
  }
int gpump_is_send_complete(struct gpump * g, int target)
  {
    return g->is_send_complete(target) ;
  }
void gpump_wait_send_complete(struct gpump * g, int target)
  {
    g->wait_send_complete(target) ;
  }
void gpump_receive(struct gpump * g, int source, struct ibv_mr * target_mr, size_t offset, size_t size)
  {
    g->receive(source, target_mr, offset, size) ;
  }
void gpump_stream_wait_recv_complete(struct gpump * g, int source, cudaStream_t stream)
  {
    g->stream_wait_recv_complete(source, stream) ;
  }
void gpump_cpu_ack_recv(struct gpump *g, int source)
  {
    g->cpu_ack_recv(source) ;
  }
int gpump_is_receive_complete(struct gpump * g, int target)
  {
    return g->is_receive_complete(target) ;
  }
void gpump_wait_receive_complete(struct gpump * g, int source)
  {
    g->wait_receive_complete(source) ;
  }
void gpump_term(struct gpump * g)
  {
    g->term() ;
    delete g ;
  }
