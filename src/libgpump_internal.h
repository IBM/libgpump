// Copyright (C) IBM Corporation 2018. All Rights Reserved
//
//    This program is licensed under the terms of the Eclipse Public License
//    v1.0 as published by the Eclipse Foundation and available at
//    http://www.eclipse.org/legal/epl-v10.html
//
//    
//    
// $COPYRIGHT$

#ifndef __libgpump_internal_h__
#define __libgpump_internal_h__
#include <mpi.h>
#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gdsync/core.h"
#include "gdsync_utils.h"
#include <vector>

using namespace std ;

struct gpump ;

class qpnumxfer
  {
public:
  int _lid ;
  uint32_t _qpnum ;
//  uint32_t _key ;
//  void * _remote_address ;
  };
class windowxfer
  {
  public:
  uint32_t _key ;
  void * _remote_address ;
  };
enum gpucomm_state {
  k_Closed ,
  k_Proposed ,
  k_Accepted
};
class gpucomm
  {
public:
  gpucomm_state _comm_state ;
  gpucomm_state _window_state ;
  struct gds_qp *_qp ;
  uint32_t _rkey ;
  qpnumxfer _xfer ;
  qpnumxfer _recv_xfer ;
  windowxfer _windowxfer ;
  windowxfer _recv_windowxfer ;
  MPI_Request _recv_send_request[2] ;
  void * _remote_address ;
  gds_wait_request_t _putgetsend_wait_request ;
  gds_wait_request_t _recv_wait_request ;
  struct ibv_mr *_local_mr ;
  struct ibv_mr *_remote_mr ;
//  void connect(int target, struct gpump * g) ;
  gpucomm() :
    _comm_state(k_Closed),
    _window_state(k_Closed),
    _qp(NULL),
    _rkey((uint32_t)-1),
    _remote_address(NULL),
    _local_mr(NULL),
    _remote_mr(NULL)
    { }
  void connect_propose(int target, struct gpump *g) ;
  void connect_accept(int target, struct gpump *g) ;
  void disconnect(void) ;
//  void create_window(int target, void * local_address, void * remote_address, size_t size, struct gpump * g) ;
  void create_window_propose(int target, void * local_address, void * remote_address, size_t size, struct gpump * g) ;
  void create_window_accept(int target, struct gpump *g) ;
  void destroy_window(void) ;
  };
class window_vector
  {
public:
    vector<gpucomm *>_gp ;
    void free() ;
    ~window_vector() ;
  };
class gpucommp
  {
public:
  gpucomm * _gp ;
  window_vector _window_vector ;
  };
struct gpump
  {
public:

  int init(MPI_Comm comm) ;
  struct ibv_mr * register_region(void * addr, size_t size) ;
  void deregister_region(struct ibv_mr * mr) ;
  void connect(int target) ;
  void connect_propose(int target) ;
  void connect_accept(int target) ;
  void disconnect(int target) ;
  void create_window(int target, void * local_address, void * remote_address, size_t size) ;
  void create_window_propose(int target, void * local_address, void * remote_address, size_t size) ;
  void replace_window_propose(int target, void * local_address, void * remote_address, size_t size) ;
  void window_accept(int target) ;
  void destroy_window(int target) ;
  void create_window_propose_x(int target, int *wx, void * local_address, void * remote_address, size_t size) ;
  void replace_window_propose_x(int target, int wx, void * local_address, void * remote_address, size_t size) ;
  void window_accept_x(int target, int wx) ;
  void destroy_window_x(int target, int wx) ;
  void cork(void) ;
  void uncork(cudaStream_t stream) ;
  void stream_put(int target, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size ) ;
  void iput(int target, size_t offset, size_t remote_offset, size_t size ) ;
  void stream_wait_put_complete(int target, cudaStream_t stream) ;
  void cpu_ack_iput(int target) ;
  bool is_put_complete(int target) ;
  void wait_put_complete(int target) ;
  void stream_get(int target, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size ) ;
  void iget(int target, size_t offset, size_t remote_offset, size_t size ) ;
  void stream_wait_get_complete(int target, cudaStream_t stream) ;
  void cpu_ack_iget(int target) ;
  bool is_get_complete(int target) ;
  void wait_get_complete(int target) ;
  void stream_put_x(int target, int wx, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size ) ;
  void iput_x(int target, int wx, size_t offset, size_t remote_offset, size_t size ) ;
  void stream_wait_put_complete_x(int target, int wx, cudaStream_t stream) ;
  void cpu_ack_iput_x(int target, int wx) ;
  bool is_put_complete_x(int target, int wx) ;
  void wait_put_complete_x(int target, int wx) ;
  void stream_get_x(int target, int wx, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size ) ;
  void iget_x(int target, int wx, size_t offset, size_t remote_offset, size_t size ) ;
  void stream_wait_get_complete_x(int target, int wx, cudaStream_t stream) ;
  void cpu_ack_iget_x(int target, int wx) ;
  bool is_get_complete_x(int target, int wx) ;
  void wait_get_complete_x(int target, int wx) ;
  void stream_send(int target, cudaStream_t stream, struct ibv_mr * source_mr, size_t offset, size_t size) ;
  void isend(int target, struct ibv_mr * source_mr, size_t offset, size_t size) ;
  void stream_wait_send_complete(int target, cudaStream_t stream) ;
  bool is_send_complete(int target) ;
  void cpu_ack_isend(int target) ;
  void wait_send_complete(int target) ;
  void receive(int source, struct ibv_mr *target_mr, size_t offset, size_t size) ;
  void stream_wait_recv_complete(int source, cudaStream_t stream) ;
  void cpu_ack_recv(int source) ;
  bool is_receive_complete(int target) ;
  void wait_receive_complete(int source) ;
  void term(void) ;

  MPI_Comm _comm ;
  int _comm_size ;
  gpucommp *_gpucomm ;
  unsigned  _window_count ;
  window_vector _window_vector ;
  int _lid ;
  struct ibv_comp_channel *_channel ;
  struct ibv_context *_ctx ;
  struct ibv_pd *_pd ;
  bool _is_corked ;
  gds_op_list_t _op_list ;
  };

#endif
