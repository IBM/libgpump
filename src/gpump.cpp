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
#include <new>
#include <iostream>
#include <iomanip>
#include <list>
#include <functional>
#include <vector>
#include <utility> //for pair
#include <map>


#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <inttypes.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "config.h"
#include "libgpump_internal.h"
#include "gdsync/core.h"
#include "gdsync_utils.h"
#include "gdsync_apis.h"
#include "gdsync_memmgr.h"
#include "gpump_cuda_wrapper.h"

using namespace std ;


enum {
  k_cqe_depth=64 ,
  k_MAX_INLINE_DATA = 208 ,
  k_DEFAULT_SQWMARK = 64 ,
  k_DEFAULT_RXWMARK = 64
};

int ib_tx_depth = 256*2;
int ib_rx_depth = 256*2;
int ib_max_sge = 30;
int ib_inline_size = 64;

struct ibv_mr * gpump::register_region(void * addr, size_t size)
  {
  gds_dbg("addr=%p, size=%lu\n",addr,size) ;
  unsigned int type ;
  CUresult curesult=cuPointerGetAttribute((void *)&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)addr);
  if ((curesult == CUDA_SUCCESS) && (type == CU_MEMORYTYPE_DEVICE)) {
       CUdeviceptr base;
       size_t size ;
       CUCHECK(cuMemGetAddressRange(&base, &size, (CUdeviceptr)addr));

       int flag = 1;
       CUCHECK(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, base));
  }

  int flags= IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  ibv_mr *ret=ibv_reg_mr(_pd, addr, size, flags) ;
  gds_dbg("ret=%p\n", (void *) ret) ;
  return ret ;
  }

void gpump::deregister_region(struct ibv_mr * mr)
  {
    ibv_dereg_mr(mr) ;
  }

int gpump::init(MPI_Comm comm)
  {
    int rc;
    _comm = comm ;
    rc = gpump_cuda_wrapper_init() ;
    if(rc) {
      return rc;
    }
    int ndevices ;
    struct ibv_device **ibvdevlist = ibv_get_device_list(&ndevices);
    if(!ibvdevlist) {
      perror("Error, ibv_get_device_list() failed");
      return 1;
    }
    if(ndevices < 1) {
      fprintf(stderr, "No IB devices found");
      return 1;
    }
    _ctx = ibv_open_device (ibvdevlist[0]);
    if(!_ctx) {
      fprintf(stderr, "Error, failed to open the device '%s'\n",
              ibv_get_device_name(ibvdevlist[0]));
      return 1;
    }
    ibv_free_device_list(ibvdevlist) ;
    _pd = ibv_alloc_pd(_ctx) ;
    if(!_pd) {
      fprintf(stderr, "Error, ibv_alloc_pd() failed\n");
      return 1;
    }
    _channel = ibv_create_comp_channel(_ctx) ;
    if(!_channel) {
      perror("Error, ibv_create_comp_channel() failed");
      return 1;
    }
//    struct ibv_srq_init_attr srq_attr;
//    memset(&srq_attr, 0, sizeof(srq_attr));
//    srq_attr.attr.max_wr  =  k_DEFAULT_RXWMARK;
//    srq_attr.attr.max_sge = 1;
//    _srq = ibv_create_srq(_pd, &srq_attr);
//    cout << "_srq=" << _srq << endl ;
    struct ibv_port_attr     portInfo;
    rc = ibv_query_port(_ctx, 1, &portInfo);
    if(rc) {
      perror("Error, ibv_query_port() failed");
      return 1;
    }
    _lid = portInfo.lid ;

//    _cq=ibv_create_cq (_ctx,
//        k_cqe_depth,
//        NULL, _channel, 0);
//    cout << "_cq=" << _cq << endl ;


    MPI_Comm_size(comm, &_comm_size) ;
    _gpucomm = new gpucommp[_comm_size] ;
    for ( int i=0; i<_comm_size; i += 1)
      {
        _gpucomm[i]._gp = NULL ;
      }
    _is_corked = false ;
    return 0;
  }

void gpump::term(void)
  {
    for ( int i=0; i<_comm_size; i += 1)
      {
        gpucomm * gp = _gpucomm[i]._gp ;
        if ( gp ) {
          gds_destroy_qp(gp -> _qp) ;
          delete gp ;
        }
      }
    delete [] _gpucomm ;

    ibv_destroy_comp_channel(_channel);
    _channel = NULL;

    ibv_dealloc_pd(_pd);
    _pd = NULL;

    ibv_close_device(_ctx);
    _ctx = NULL;
//    if ( _local_mr ) deregister_region(_local_mr) ;
//    if ( _remote_mr ) deregister_region(_remote_mr) ;
  }

void gpucomm::connect_propose(int target, struct gpump * g)
  {
    if ( _comm_state == k_Closed)
      {
        gds_qp_init_attr_t ib_qp_init_attr;
        memset(&ib_qp_init_attr, 0, sizeof(ib_qp_init_attr));
        ib_qp_init_attr.cap.max_send_wr  = ib_tx_depth;
        ib_qp_init_attr.cap.max_recv_wr  = ib_rx_depth;
        ib_qp_init_attr.cap.max_send_sge = ib_max_sge;
        ib_qp_init_attr.cap.max_recv_sge = ib_max_sge;
        ib_qp_init_attr.qp_type = IBV_QPT_RC;
        ib_qp_init_attr.cap.max_inline_data = ib_inline_size;
        int gds_flags = GDS_CREATE_QP_DEFAULT;
        _qp = gds_create_qp(g->_pd, g->_ctx, &ib_qp_init_attr, 0, gds_flags);
        if ( _qp == NULL ) {
          perror("Error, gds_create_qp() failed");
          abort();
        }
    //    qpnumxfer recv_xfer ;
    //    MPI_Request request ;
        MPI_Irecv(&_recv_xfer, sizeof(_recv_xfer), MPI_CHAR, target, 0, g->_comm, _recv_send_request+0) ;
    //    qpnumxfer xfer ;
        _xfer._lid = g->_lid ;
        _xfer._qpnum = _qp->qp->qp_num ;
    //    if ( g->_remote_mr) {
    //      xfer._key = g->_remote_mr->rkey ;
    //      xfer._remote_address = g->_remote_mr->addr ;
    //    }
    //    else {
    //      xfer._key = 0 ;
    //      xfer._remote_address = NULL ;
    //    }
        MPI_Isend(&_xfer, sizeof(_xfer), MPI_CHAR, target, 0, g->_comm, _recv_send_request+1) ;
        _comm_state = k_Proposed ;
      }
  }
void gpucomm::connect_accept(int target, struct gpump * g)
  {
    if ( _comm_state == k_Proposed)
      {
        MPI_Status status[2] ;
        MPI_Waitall(2, _recv_send_request, status) ;
        assert(0 == status[0].MPI_ERROR) ;
        assert(0 == status[1].MPI_ERROR) ;
    //    _rkey = recv_xfer._key ;
    //    _remote_address = recv_xfer._remote_address ;
        struct ibv_exp_qp_attr attr;
        // Set qpairs to INIT
        memset(&attr, 0, sizeof(attr));
        attr.qp_state        = IBV_QPS_INIT;
        attr.port_num        = 1 ;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                    IBV_ACCESS_REMOTE_ATOMIC;
        {
          int rc=ibv_exp_modify_qp(_qp->qp, &attr,
                                      IBV_EXP_QP_STATE          |
                                      IBV_EXP_QP_PKEY_INDEX     |
                                      IBV_EXP_QP_PORT           |
                                      IBV_EXP_QP_ACCESS_FLAGS) ;
        }
        // Bring in the remote queue pair number, set qp to RTR
        memset(&attr, 0, sizeof(attr));
        attr.qp_state         = IBV_QPS_RTR;
        attr.path_mtu         = IBV_MTU_4096;
        attr.rq_psn                 = 0;
        attr.max_dest_rd_atomic     = 16;
        attr.min_rnr_timer          = 12 ;
        attr.ah_attr.is_global      = 0 ;
        attr.ah_attr.sl         =  0 ;
        attr.ah_attr.src_path_bits  = 0;
        attr.ah_attr.port_num       = 1 ;

        uint64_t modify_attr_mask_RTR =
          IBV_QP_STATE              |
          IBV_QP_AV                 |
          IBV_QP_DEST_QPN           |
          IBV_QP_PATH_MTU           |
          IBV_QP_RQ_PSN             |
          IBV_QP_MAX_DEST_RD_ATOMIC |
          IBV_QP_MIN_RNR_TIMER
    //      |      IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
          ;

        attr.ah_attr.dlid           = _recv_xfer._lid ;
        attr.dest_qp_num      =_recv_xfer._qpnum;
        {
          int rc=ibv_exp_modify_qp(_qp->qp, &attr, modify_attr_mask_RTR) ;
        }
        // Set QP to RTS
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS ;
        attr.timeout          = 20 ;
        attr.retry_cnt        = 7 ;
        attr.rnr_retry        = 7 ;
        attr.sq_psn           = 0;
        attr.max_rd_atomic    = 16 ;

        {
          int rc=ibv_exp_modify_qp(_qp->qp, &attr,
                  IBV_QP_STATE              |
                  IBV_QP_TIMEOUT            |
                  IBV_QP_RETRY_CNT          |
                  IBV_QP_RNR_RETRY          |
                  IBV_QP_SQ_PSN             |
                  IBV_QP_MAX_QP_RD_ATOMIC
              ) ;
        }
    //        CUDA_CHECK(cudaStreamCreate(&_stream)) ;
        _comm_state = k_Accepted ;
      }

  }

void gpucomm::disconnect(void)
  {
    assert(_comm_state == k_Accepted) ;
    assert(_window_state == k_Closed) ;
    gds_destroy_qp(_qp) ;
    _comm_state = k_Closed ;
  }

window_vector::~window_vector() { free() ; }


void window_vector::free()
  {
    for ( int x=0; x<_gp.size(); x+=1)
      {
        if ( _gp[x] != NULL )
          {
            _gp[x]->destroy_window() ;
            delete _gp[x] ;
            _gp[x] = NULL ;
          }
      }
  }


void gpump::connect(int target)
  {
    gds_dbg("(Deprecated) Connecting to rank=%d\n", target) ;
    assert(target < _comm_size) ;
//    int comm_rank ;
//    MPI_Comm_rank(_comm, &comm_rank) ;
//    assert(target != comm_rank) ; // Self-connect not supported ...
    if ( _gpucomm[target]._gp == NULL)
      {
        _gpucomm[target]._gp = new gpucomm ;
        _gpucomm[target]._gp->connect_propose(target, this) ;
        _gpucomm[target]._gp->connect_accept(target, this) ;
      }
  }

void gpump::connect_propose(int target)
  {
    gds_dbg("Proposing connection to rank=%d\n", target) ;
    assert(target < _comm_size) ;
//    int comm_rank ;
//    MPI_Comm_rank(_comm, &comm_rank) ;
//    assert(target != comm_rank) ; // Self-connect not supported ...
    if ( _gpucomm[target]._gp == NULL)
      {
        _gpucomm[target]._gp = new gpucomm ;
        _gpucomm[target]._gp->connect_propose(target, this) ;
      }
  }
void gpump::connect_accept(int target)
  {
    gds_dbg("Accepting connection to rank=%d\n", target) ;
    assert(target < _comm_size) ;
    _gpucomm[target]._gp->connect_accept(target, this) ;
  }

void gpump::disconnect(int target)
  {
    gds_dbg("Disconnecting from rank=%d\n", target) ;
    assert(target < _comm_size) ;
    if ( _gpucomm[target]._gp != NULL )
      {
        _gpucomm[target]._gp->disconnect() ;
        delete _gpucomm[target]._gp ;
        _gpucomm[target]._gp = NULL ;
      }
  }

//-----------------------------------------------------------------------------

void gpucomm::destroy_window(void)
  {
    assert(_window_state == k_Accepted) ;
    ibv_dereg_mr(_local_mr) ;
    ibv_dereg_mr(_remote_mr) ;
    _window_state = k_Closed ;
  }
void gpucomm::create_window_propose(int target, void * local_address, void * remote_address, size_t size, struct gpump * g  )
  {
    assert(_comm_state == k_Accepted) ;
    assert(_window_state == k_Closed) ;
    int flags= IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
    _local_mr = ibv_reg_mr(g->_pd, local_address, size, flags) ;
    _remote_mr = ibv_reg_mr(g->_pd, remote_address, size, flags) ;
//    windowxfer recv_xfer ;
//    MPI_Request request ;
    MPI_Irecv(&_recv_windowxfer, sizeof(_recv_windowxfer), MPI_CHAR, target, 0, g->_comm, _recv_send_request+0) ;
//    windowxfer xfer ;
    _windowxfer._key = _remote_mr->rkey ;
    _windowxfer._remote_address = _remote_mr->addr ;
    MPI_Isend(&_windowxfer, sizeof(_windowxfer), MPI_CHAR, target, 0, g->_comm, _recv_send_request+1) ;
    _window_state = k_Proposed ;
  }
void gpucomm::create_window_accept(int target, struct gpump *g)
  {
    assert(_window_state == k_Proposed) ;
    MPI_Status status[2] ;
    MPI_Waitall(2, _recv_send_request, status) ;
    assert(0 == status[0].MPI_ERROR) ;
    assert(0 == status[1].MPI_ERROR) ;
    _rkey = _recv_windowxfer._key ;
    _remote_address = _recv_windowxfer._remote_address ;
    _window_state = k_Accepted ;

  }
void gpump::destroy_window(int target)
  {
    gds_dbg("Destroying RDMA window to rank=%d\n", target) ;
    _gpucomm[target]._gp->destroy_window() ;
  }
void gpump::destroy_window_x(int target, int wx)
  {
    gds_dbg("Destroying RDMA window index=%d to rank=%d\n", wx, target) ;
    _gpucomm[target]._window_vector._gp[wx]->destroy_window() ;
    delete _gpucomm[target]._window_vector._gp[wx] ;
    _gpucomm[target]._window_vector._gp[wx] = NULL ;
  }
void gpump::create_window_propose(int target, void * local_address, void * remote_address, size_t size)
  {
    gds_dbg("Proposing creation of RDMA window to rank=%d local_address=%p remote_address=%p size=%lu\n",
             target,local_address,remote_address, size) ;
    _gpucomm[target]._gp->create_window_propose(target, local_address, remote_address,size, this) ;
  }
void gpump::replace_window_propose(int target, void * local_address, void * remote_address, size_t size)
  {
    gds_dbg("Proposing creation of RDMA window to rank=%d local_address=%p remote_address=%p size=%lu\n",
             target,local_address,remote_address, size) ;
    _gpucomm[target]._gp->destroy_window() ;
    _gpucomm[target]._gp->create_window_propose(target, local_address, remote_address,size, this) ;
  }
void gpump::window_accept(int target)
  {
    gds_dbg("Accepting creation of RDMA window to rank=%d\n", target) ;
    _gpucomm[target]._gp->create_window_accept(target, this) ;
  }
void gpump::create_window(int target, void * local_address, void * remote_address, size_t size)
  {
    gds_dbg("(Deprecated) Creating RDMA window to rank=%d local_address=%p remote_address=%p size=%lu\n",
             target,local_address,remote_address, size) ;
    _gpucomm[target]._gp->create_window_propose(target, local_address, remote_address,size, this) ;
    _gpucomm[target]._gp->create_window_accept(target, this) ;
  }
void gpump::create_window_propose_x(int target, int *wx, void * local_address, void * remote_address, size_t size)
  {
    gds_dbg("Proposing creation of RDMA window to rank=%d local_address=%p remote_address=%p size=%lu\n",
        target,local_address,remote_address, size) ;
    *wx = _gpucomm[target]._window_vector._gp.size() ;
    gpucomm *q=new gpucomm ;
    q->create_window_propose(target, local_address, remote_address,size, this) ;
    _gpucomm[target]._window_vector._gp.push_back(q) ;
  }
void gpump::replace_window_propose_x(int target, int wx, void * local_address, void * remote_address, size_t size)
  {
    gds_dbg("Proposing replacement of RDMA window index %d to rank=%d local_address=%p remote_address=%p size=%lu\n",
        wx, target,local_address,remote_address, size) ;
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    _gpucomm[target]._window_vector._gp[wx]->destroy_window() ;
    _gpucomm[target]._window_vector._gp[wx]->create_window_propose(target, local_address, remote_address,size, this) ;
  }
void gpump::window_accept_x(int target, int wx)
  {
    gds_dbg("Accepting  RDMA window index=%d to rank=%d\n", wx, target) ;
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    _gpucomm[target]._window_vector._gp[wx]->create_window_accept(target, this) ;
  }
void gpump::cork(void)
  {
    _is_corked = true ;
  }
void gpump::uncork(cudaStream_t stream)
  {
    assert(_is_corked) ;
    _is_corked = false ;
    gds_uncork_post_descriptors(stream, _op_list) ;
  }
void gpump::stream_put(int target, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size)
  {
    struct ibv_sge sg_entry ;
    sg_entry.lkey = _gpucomm[target]._gp->_local_mr->lkey ;
    sg_entry.addr = ((uint64_t) (_gpucomm[target]._gp->_local_mr->addr)) + offset ;
    sg_entry.length = size ;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL ;
    rr.num_sge = 1 ;
    rr.sg_list = &sg_entry ;
    rr.wr_id = 0 ;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_RDMA_WRITE;
    rr.wr.rdma.remote_addr = ((uint64_t) _gpucomm[target]._gp->_remote_address) + remote_offset ;
    rr.wr.rdma.rkey = _gpucomm[target]._gp->_rkey;
    struct ibv_exp_send_wr *bad_rr ;
    int ret=ibv_exp_post_send(_gpucomm[target]._gp->_qp->qp, &rr, &bad_rr) ;
    if ( ret) {
       gds_err("error %d in ibv_exp_post_send\n", ret) ;
       goto out ;
    }
    gds_init_send_info(&request) ;
    ibv_exp_peer_commit_qp(_gpucomm[target]._gp->_qp->qp, &request.commit);
    gds_descriptor_t descs[1];
    descs[0].tag = GDS_TAG_SEND;
    descs[0].send = &request;

    if ( _is_corked)
      {
        ret=gds_cork_post_descriptors(stream, _op_list, 1, descs, 0) ;
        if (ret) {
                gds_err("error %d in gds_cork_post_descriptors\n", ret);
                goto out;
        }
      }
    else
      {
        ret=gds_stream_post_descriptors(stream, 1, descs, 0);
        if (ret) {
                gds_err("error %d in gds_stream_post_descriptors\n", ret);
                goto out;
        }
      }
    ret = gds_prepare_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;
//    return ret;

  }
void gpump::stream_put_x(int target, int wx, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    struct ibv_sge sg_entry ;
    sg_entry.lkey = _gpucomm[target]._window_vector._gp[wx]->_local_mr->lkey ;
    sg_entry.addr = ((uint64_t) (_gpucomm[target]._window_vector._gp[wx]->_local_mr->addr)) + offset ;
    sg_entry.length = size ;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL ;
    rr.num_sge = 1 ;
    rr.sg_list = &sg_entry ;
    rr.wr_id = 0 ;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_RDMA_WRITE;
    rr.wr.rdma.remote_addr = ((uint64_t) _gpucomm[target]._window_vector._gp[wx]->_remote_address) + remote_offset ;
    rr.wr.rdma.rkey = _gpucomm[target]._gp->_rkey;
    struct ibv_exp_send_wr *bad_rr ;
    int ret=ibv_exp_post_send(_gpucomm[target]._window_vector._gp[wx]->_qp->qp, &rr, &bad_rr) ;
    if ( ret) {
       gds_err("error %d in ibv_exp_post_send\n", ret) ;
       goto out ;
    }
    gds_init_send_info(&request) ;
    ibv_exp_peer_commit_qp(_gpucomm[target]._window_vector._gp[wx]->_qp->qp, &request.commit);
    gds_descriptor_t descs[1];
    descs[0].tag = GDS_TAG_SEND;
    descs[0].send = &request;

    if ( _is_corked)
      {
        ret=gds_cork_post_descriptors(stream, _op_list, 1, descs, 0) ;
        if (ret) {
                gds_err("error %d in gds_cork_post_descriptors\n", ret);
                goto out;
        }
      }
    else
      {
        ret=gds_stream_post_descriptors(stream, 1, descs, 0);
        if (ret) {
                gds_err("error %d in gds_stream_post_descriptors\n", ret);
                goto out;
        }
      }
    ret = gds_prepare_wait_cq(&_gpucomm[target]._window_vector._gp[wx]->_qp->send_cq,
                              &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;
//    return ret;

  }

void gpump::iput(int target, size_t offset, size_t remote_offset, size_t size)
  {
    struct ibv_sge sg_entry ;
    sg_entry.lkey = _gpucomm[target]._gp->_local_mr->lkey ;
    sg_entry.addr = ((uint64_t) (_gpucomm[target]._gp->_local_mr->addr)) + offset ;
    sg_entry.length = size ;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL ;
    rr.num_sge = 1 ;
    rr.sg_list = &sg_entry ;
    rr.wr_id = 0 ;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_RDMA_WRITE;
    rr.wr.rdma.remote_addr = ((uint64_t) _gpucomm[target]._gp->_remote_address) + remote_offset ;
    rr.wr.rdma.rkey = _gpucomm[target]._gp->_rkey;
    struct ibv_exp_send_wr *bad_rr ;
//    int ret=ibv_exp_post_send(_gpucomm[target]._gp->_qp->qp, &rr, &bad_rr) ;
//    if ( ret) {
//       gds_err("error %d in ibv_exp_post_send\n", ret) ;
//       goto out ;
//    }
//    gds_init_send_info(&request) ;
//    ibv_exp_peer_commit_qp(_gpucomm[target]._gp->_qp->qp, &request.commit);
//    gds_descriptor_t descs[1];
//    descs[0].tag = GDS_TAG_SEND;
//    descs[0].send = &request;
//
//    ret=gds_stream_post_descriptors(stream, 1, descs, 0);
    int ret = gds_post_send (_gpucomm[target]._gp->_qp, &rr, &bad_rr);
    if (ret) {
            gds_err("error %d in gds_post_send\n", ret);
            goto out;
    }
    ret = gds_prepare_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;
//    return ret;

  }
void gpump::iput_x(int target, int wx, size_t offset, size_t remote_offset, size_t size)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    struct ibv_sge sg_entry ;
    sg_entry.lkey = _gpucomm[target]._window_vector._gp[wx]->_local_mr->lkey ;
    sg_entry.addr = ((uint64_t) (_gpucomm[target]._window_vector._gp[wx]->_local_mr->addr)) + offset ;
    sg_entry.length = size ;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL ;
    rr.num_sge = 1 ;
    rr.sg_list = &sg_entry ;
    rr.wr_id = 0 ;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_RDMA_WRITE;
    rr.wr.rdma.remote_addr = ((uint64_t) _gpucomm[target]._window_vector._gp[wx]->_remote_address) + remote_offset ;
    rr.wr.rdma.rkey = _gpucomm[target]._gp->_rkey;
    struct ibv_exp_send_wr *bad_rr ;
//    int ret=ibv_exp_post_send(_gpucomm[target]._gp->_qp->qp, &rr, &bad_rr) ;
//    if ( ret) {
//       gds_err("error %d in ibv_exp_post_send\n", ret) ;
//       goto out ;
//    }
//    gds_init_send_info(&request) ;
//    ibv_exp_peer_commit_qp(_gpucomm[target]._gp->_qp->qp, &request.commit);
//    gds_descriptor_t descs[1];
//    descs[0].tag = GDS_TAG_SEND;
//    descs[0].send = &request;
//
//    ret=gds_stream_post_descriptors(stream, 1, descs, 0);
    int ret = gds_post_send (_gpucomm[target]._window_vector._gp[wx]->_qp, &rr, &bad_rr);
    if (ret) {
            gds_err("error %d in gds_post_send\n", ret);
            goto out;
    }
    ret = gds_prepare_wait_cq(&_gpucomm[target]._window_vector._gp[wx]->_qp->send_cq,
                              &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;
//    return ret;

  }

void gpump::stream_send(int target, cudaStream_t stream, struct ibv_mr * source_mr, size_t offset, size_t size)
  {
    gds_dbg("stream_send(target=%d, addr=%p, offset=%ld, size=%ld)\n",target, source_mr->addr, offset, size) ;
//    assert(offset >= 0) ;
    assert(offset+size <= source_mr->length) ;
    struct ibv_sge sg_entry ;
    sg_entry.lkey = source_mr->lkey ;
    sg_entry.addr = ((uint64_t) (source_mr->addr)) + offset ;
    sg_entry.length = size;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL ;
    rr.num_sge = 1 ;
    rr.sg_list = &sg_entry ;
    rr.wr_id = 0 ;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_SEND;
    struct ibv_exp_send_wr *bad_rr ;
    int ret=ibv_exp_post_send(_gpucomm[target]._gp->_qp->qp, &rr, &bad_rr) ;
    if ( ret) {
       gds_err("error %d in ibv_exp_post_send\n", ret) ;
       goto out ;
    }
    gds_init_send_info(&request) ;
    ibv_exp_peer_commit_qp(_gpucomm[target]._gp->_qp->qp, &request.commit);
    gds_descriptor_t descs[1];
    descs[0].tag = GDS_TAG_SEND;
    descs[0].send = &request;

    if ( _is_corked)
      {
        ret=gds_cork_post_descriptors(stream, _op_list, 1, descs, 0) ;
        if (ret) {
                gds_err("error %d in gds_cork_post_descriptors\n", ret);
                goto out;
        }
      }
    else
      {
        ret=gds_stream_post_descriptors(stream, 1, descs, 0);
        if (ret) {
                gds_err("error %d in gds_stream_post_descriptors\n", ret);
                goto out;
        }
      }
    ret = gds_prepare_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;
//    return ret;

  }

void gpump::isend(int target, struct ibv_mr * source_mr, size_t offset, size_t size)
  {
    gds_dbg("stream_send(target=%d, addr=%p, offset=%ld, size=%ld)\n",target, source_mr->addr, offset, size) ;
//    assert(offset >= 0) ;
    assert(offset+size <= source_mr->length) ;
    struct ibv_sge sg_entry ;
    sg_entry.lkey = source_mr->lkey ;
    sg_entry.addr = ((uint64_t) (source_mr->addr)) + offset ;
    sg_entry.length = size;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL ;
    rr.num_sge = 1 ;
    rr.sg_list = &sg_entry ;
    rr.wr_id = 0 ;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_SEND;
    struct ibv_exp_send_wr *bad_rr ;
//    int ret=ibv_exp_post_send(_gpucomm[target]._gp->_qp->qp, &rr, &bad_rr) ;
//    if ( ret) {
//       gds_err("error %d in ibv_exp_post_send\n", ret) ;
//       goto out ;
//    }
//    gds_init_send_info(&request) ;
//    ibv_exp_peer_commit_qp(_gpucomm[target]._gp->_qp->qp, &request.commit);
//    gds_descriptor_t descs[1];
//    descs[0].tag = GDS_TAG_SEND;
//    descs[0].send = &request;

    int ret = gds_post_send (_gpucomm[target]._gp->_qp, &rr, &bad_rr);
    if (ret) {
            gds_err("error %d in gds_post_send\n", ret);
            goto out;
    }
    ret = gds_prepare_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;
//    return ret;

  }

void gpump::receive(int source, struct ibv_mr * target_mr, size_t offset, size_t size)
  {
    gds_dbg("receive(source=%d, addr=%p, offset=%ld, size=%ld)\n",source, target_mr->addr, offset, size) ;
//    assert(offset >= 0) ;
    assert(offset+size <= target_mr->length) ;
    struct ibv_sge sg_entry ;
    sg_entry.lkey = target_mr->lkey ;
    sg_entry.addr = ((uint64_t) (target_mr->addr)) + offset ;
    sg_entry.length = size;
    struct ibv_recv_wr wr ;
    gds_send_request_t request ;
    wr.next = NULL ;
    wr.num_sge = 1 ;
    wr.sg_list = &sg_entry ;
    wr.wr_id = 0 ;
    struct ibv_recv_wr *bad_wr ;
    int ret=ibv_post_recv(_gpucomm[source]._gp->_qp->qp, &wr, &bad_wr) ;
    if ( ret) {
       gds_err("error %d in ibv_post_recv\n", ret) ;
       goto out ;
    }

    ret = gds_prepare_wait_cq(&_gpucomm[source]._gp->_qp->recv_cq, &_gpucomm[source]._gp->_recv_wait_request, 0);
    if (ret) {
       gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
       // BUG: leaking req ??
       goto out;
    }

    out: ;
//    return ret;

  }

void gpump::stream_wait_put_complete(int target, cudaStream_t stream)
  {
    if ( _is_corked )
      {
        gds_cork_post_wait_cq(stream, _op_list, &_gpucomm[target]._gp->_putgetsend_wait_request) ;
      }
    else
      {
        gds_stream_post_wait_cq(stream, &_gpucomm[target]._gp->_putgetsend_wait_request) ;
      }
  }

void gpump::stream_wait_put_complete_x(int target, int wx, cudaStream_t stream)
  {
    if ( _is_corked )
      {
        gds_cork_post_wait_cq(stream, _op_list, &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request) ;
      }
    else
      {
        gds_stream_post_wait_cq(stream, &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request) ;
      }
  }

void gpump::cpu_ack_iput(int target)
  {
    int ret = gds_post_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
  }

void gpump::cpu_ack_iput_x(int target, int wx)
  {
    int ret = gds_post_wait_cq(&_gpucomm[target]._window_vector._gp[wx]->_qp->send_cq,
                               &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request, 0);
  }

void gpump::stream_wait_send_complete(int target, cudaStream_t stream)
  {
    if ( _is_corked )
      {
        gds_cork_post_wait_cq(stream, _op_list, &_gpucomm[target]._gp->_putgetsend_wait_request) ;
      }
    else
      {
        gds_stream_post_wait_cq(stream, &_gpucomm[target]._gp->_putgetsend_wait_request) ;
      }
  }

void gpump::cpu_ack_isend(int target)
  {
    int ret = gds_post_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
  }

void gpump::stream_wait_recv_complete(int source, cudaStream_t stream)
  {
    if ( _is_corked )
      {
        gds_cork_post_wait_cq(stream, _op_list, &_gpucomm[source]._gp->_recv_wait_request) ;
      }
    else
      {
        gds_stream_post_wait_cq(stream, &_gpucomm[source]._gp->_recv_wait_request) ;
      }
  }

void gpump::cpu_ack_recv(int source)
  {
    int ret = gds_post_wait_cq(&_gpucomm[source]._gp->_qp->recv_cq, &_gpucomm[source]._gp->_recv_wait_request, 0);
  }

bool gpump::is_put_complete(int target)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._gp->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( ne > 0 )
      {
        if ( wc.status != IBV_WC_SUCCESS)
          {
            gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
            abort() ;
          }
        return true ;
      }
    else return false ;
  }

bool gpump::is_put_complete_x(int target, int wx)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._window_vector._gp[wx]->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( ne > 0 )
      {
        if ( wc.status != IBV_WC_SUCCESS)
          {
            gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
            abort() ;
          }
        return true ;
      }
    else return false ;
  }

void gpump::wait_put_complete(int target)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._gp->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    while ( 0 == ne ) ne = ibv_poll_cq(cq, 1, &wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( wc.status != IBV_WC_SUCCESS)
      {
        gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
        abort() ;
      }
  }

void gpump::wait_put_complete_x(int target, int wx)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._window_vector._gp[wx]->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    while ( 0 == ne ) ne = ibv_poll_cq(cq, 1, &wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( wc.status != IBV_WC_SUCCESS)
      {
        gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
        abort() ;
      }
  }

bool gpump::is_send_complete(int target)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._gp->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( ne > 0 )
      {
        if ( wc.status != IBV_WC_SUCCESS)
          {
            gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
            abort() ;
          }
        return true ;
      }
    else return false ;
  }

void gpump::wait_send_complete(int target)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._gp->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    while ( 0 == ne ) ne=ibv_poll_cq(cq, 1, &wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( wc.status != IBV_WC_SUCCESS)
      {
        gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
        abort() ;
      }
  }

bool gpump::is_receive_complete(int target)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._gp->_qp->recv_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( ne > 0 )
      {
        if ( wc.status != IBV_WC_SUCCESS)
          {
            gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
            abort() ;
          }
        return true ;
      }
    else return false ;
  }

void gpump::wait_receive_complete(int source)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[source]._gp->_qp->recv_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    while ( 0 == ne ) ne=ibv_poll_cq(cq, 1, &wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( wc.status != IBV_WC_SUCCESS)
      {
        gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
        abort() ;
      }
  }

void gpump::stream_get(int target, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size)
  {
    struct ibv_sge sg_entry ;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_RDMA_READ;
    rr.wr_id = 0;
    rr.num_sge = 1;
    rr.sg_list = &sg_entry;

    sg_entry.length = size;
    sg_entry.lkey = _gpucomm[target]._gp->_local_mr->lkey;
    sg_entry.addr = ((uint64_t) (_gpucomm[target]._gp->_local_mr->addr)) + offset;

    rr.wr.rdma.remote_addr = ((uint64_t) _gpucomm[target]._gp->_remote_address) + remote_offset ;
    rr.wr.rdma.rkey = _gpucomm[target]._gp->_rkey;

    struct ibv_exp_send_wr *bad_rr ;
    int ret=ibv_exp_post_send(_gpucomm[target]._gp->_qp->qp, &rr, &bad_rr) ;
    if ( ret) {
       gds_err("error %d in ibv_exp_post_send\n", ret) ;
       goto out ;
    }
    gds_init_send_info(&request) ;
    ibv_exp_peer_commit_qp(_gpucomm[target]._gp->_qp->qp, &request.commit);
    gds_descriptor_t descs[1];
    descs[0].tag = GDS_TAG_SEND;
    descs[0].send = &request;

    if ( _is_corked)
      {
        ret=gds_cork_post_descriptors(stream, _op_list, 1, descs, 0) ;
        if (ret) {
                gds_err("error %d in gds_cork_post_descriptors\n", ret);
                goto out;
        }
      }
    else
      {
        ret=gds_stream_post_descriptors(stream, 1, descs, 0);
        if (ret) {
                gds_err("error %d in gds_stream_post_descriptors\n", ret);
                goto out;
        }
      }
    ret = gds_prepare_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;
  }

void gpump::stream_get_x(int target, int wx, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    struct ibv_sge sg_entry ;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_RDMA_READ;
    rr.wr_id = 0;
    rr.num_sge = 1;
    rr.sg_list = &sg_entry;

    sg_entry.length = size;
    sg_entry.lkey = _gpucomm[target]._window_vector._gp[wx]->_local_mr->lkey;
    sg_entry.addr = ((uint64_t) (_gpucomm[target]._window_vector._gp[wx]->_local_mr->addr)) + offset;

    rr.wr.rdma.remote_addr = ((uint64_t) _gpucomm[target]._window_vector._gp[wx]->_remote_address) + remote_offset ;
    rr.wr.rdma.rkey = _gpucomm[target]._window_vector._gp[wx]->_rkey;

    struct ibv_exp_send_wr *bad_rr ;
    int ret=ibv_exp_post_send(_gpucomm[target]._window_vector._gp[wx]->_qp->qp, &rr, &bad_rr) ;
    if ( ret) {
       gds_err("error %d in ibv_exp_post_send\n", ret) ;
       goto out ;
    }
    gds_init_send_info(&request) ;
    ibv_exp_peer_commit_qp(_gpucomm[target]._window_vector._gp[wx]->_qp->qp, &request.commit);
    gds_descriptor_t descs[1];
    descs[0].tag = GDS_TAG_SEND;
    descs[0].send = &request;

    if ( _is_corked)
      {
        ret=gds_cork_post_descriptors(stream, _op_list, 1, descs, 0) ;
        if (ret) {
                gds_err("error %d in gds_cork_post_descriptors\n", ret);
                goto out;
        }
      }
    else
      {
        ret=gds_stream_post_descriptors(stream, 1, descs, 0);
        if (ret) {
                gds_err("error %d in gds_stream_post_descriptors\n", ret);
                goto out;
        }
      }
    ret = gds_prepare_wait_cq(&_gpucomm[target]._window_vector._gp[wx]->_qp->send_cq,
                              &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;
  }


void gpump::iget(int target, size_t offset, size_t remote_offset, size_t size)
  {
    struct ibv_sge sg_entry ;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_RDMA_READ;
    rr.wr_id = 0;
    rr.num_sge = 1;
    rr.sg_list = &sg_entry;

    sg_entry.length = size;
    sg_entry.lkey = _gpucomm[target]._gp->_local_mr->lkey;
    sg_entry.addr = ((uint64_t) (_gpucomm[target]._gp->_local_mr->addr)) + offset;

    rr.wr.rdma.remote_addr = ((uint64_t) _gpucomm[target]._gp->_remote_address) + remote_offset ;
    rr.wr.rdma.rkey = _gpucomm[target]._gp->_rkey;

    struct ibv_exp_send_wr *bad_rr ;
//    int ret=ibv_exp_post_send(_gpucomm[target]._gp->_qp->qp, &rr, &bad_rr) ;
//    if ( ret) {
//       gds_err("error %d in ibv_exp_post_send\n", ret) ;
//       goto out ;
//    }
//    gds_init_send_info(&request) ;
//    ibv_exp_peer_commit_qp(_gpucomm[target]._gp->_qp->qp, &request.commit);
//    gds_descriptor_t descs[1];
//    descs[0].tag = GDS_TAG_SEND;
//    descs[0].send = &request;

    int ret = gds_post_send (_gpucomm[target]._gp->_qp, &rr, &bad_rr);
    if (ret) {
            gds_err("error %d in gds_post_send\n", ret);
            goto out;
    }

    ret = gds_prepare_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;

  }
void gpump::iget_x(int target, int wx, size_t offset, size_t remote_offset, size_t size)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    struct ibv_sge sg_entry ;
    struct ibv_exp_send_wr rr ;
    gds_send_request_t request ;
    rr.next = NULL;
    rr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    rr.exp_opcode = IBV_EXP_WR_RDMA_READ;
    rr.wr_id = 0;
    rr.num_sge = 1;
    rr.sg_list = &sg_entry;

    sg_entry.length = size;
    sg_entry.lkey = _gpucomm[target]._window_vector._gp[wx]->_local_mr->lkey;
    sg_entry.addr = ((uint64_t) (_gpucomm[target]._window_vector._gp[wx]->_local_mr->addr)) + offset;

    rr.wr.rdma.remote_addr = ((uint64_t) _gpucomm[target]._window_vector._gp[wx]->_remote_address) + remote_offset ;
    rr.wr.rdma.rkey = _gpucomm[target]._window_vector._gp[wx]->_rkey;

    struct ibv_exp_send_wr *bad_rr ;
//    int ret=ibv_exp_post_send(_gpucomm[target]._gp->_qp->qp, &rr, &bad_rr) ;
//    if ( ret) {
//       gds_err("error %d in ibv_exp_post_send\n", ret) ;
//       goto out ;
//    }
//    gds_init_send_info(&request) ;
//    ibv_exp_peer_commit_qp(_gpucomm[target]._gp->_qp->qp, &request.commit);
//    gds_descriptor_t descs[1];
//    descs[0].tag = GDS_TAG_SEND;
//    descs[0].send = &request;

    int ret = gds_post_send (_gpucomm[target]._window_vector._gp[wx]->_qp, &rr, &bad_rr);
    if (ret) {
            gds_err("error %d in gds_post_send\n", ret);
            goto out;
    }

    ret = gds_prepare_wait_cq(&_gpucomm[target]._window_vector._gp[wx]->_qp->send_cq,
        &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request, 0);
    if (ret) {
        gds_err("gds_prepare_wait_cq failed: %s \n", strerror(ret));
        // BUG: leaking req ??
        goto out;
    }

    out: ;

  }

void gpump::stream_wait_get_complete(int target, cudaStream_t stream)
  {
    if ( _is_corked )
      {
        gds_cork_post_wait_cq(stream, _op_list, &_gpucomm[target]._gp->_putgetsend_wait_request) ;
      }
    else
      {
        gds_stream_post_wait_cq(stream, &_gpucomm[target]._gp->_putgetsend_wait_request) ;
      }
  }

void gpump::stream_wait_get_complete_x(int target, int wx, cudaStream_t stream)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    if ( _is_corked )
      {
        gds_cork_post_wait_cq(stream, _op_list, &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request) ;
      }
    else
      {
        gds_stream_post_wait_cq(stream, &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request) ;
      }
  }

void gpump::cpu_ack_iget(int target)
  {
    int ret = gds_post_wait_cq(&_gpucomm[target]._gp->_qp->send_cq, &_gpucomm[target]._gp->_putgetsend_wait_request, 0);
  }

void gpump::cpu_ack_iget_x(int target, int wx)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    int ret = gds_post_wait_cq(&_gpucomm[target]._window_vector._gp[wx]->_qp->send_cq,
                               &_gpucomm[target]._window_vector._gp[wx]->_putgetsend_wait_request, 0);
  }

bool gpump::is_get_complete(int target)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._gp->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( ne > 0 )
      {
        if ( wc.status != IBV_WC_SUCCESS)
          {
            gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
            abort() ;
          }
        return true ;
      }
    else return false ;
  }

bool gpump::is_get_complete_x(int target, int wx)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._window_vector._gp[wx]->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( ne > 0 )
      {
        if ( wc.status != IBV_WC_SUCCESS)
          {
            gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
            abort() ;
          }
        return true ;
      }
    else return false ;
  }

void gpump::wait_get_complete(int target)
  {
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._gp->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    while ( 0 == ne ) ne = ibv_poll_cq(cq, 1, &wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( wc.status != IBV_WC_SUCCESS)
      {
        gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
        abort() ;
      }
  }

void gpump::wait_get_complete_x(int target, int wx)
  {
    assert(wx < _gpucomm[target]._window_vector._gp.size()) ;
    struct ibv_wc wc ;
    struct ibv_cq * cq = _gpucomm[target]._window_vector._gp[wx]->_qp->send_cq.cq ;
    int ne=ibv_poll_cq(cq,1,&wc) ;
    while ( 0 == ne ) ne = ibv_poll_cq(cq, 1, &wc) ;
    if ( ne < 0 )
      {
        gds_err("error %d(%d) in ibv_poll_cq\n", ne, errno);
        abort() ;
      }
    if ( wc.status != IBV_WC_SUCCESS)
      {
        gds_err("status=%d in ibv_poll_cq\n", wc.status) ;
        abort() ;
      }
  }

//-----------------------------------------------------------------------------


