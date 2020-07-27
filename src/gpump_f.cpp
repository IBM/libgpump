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
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
using namespace std ;

static gpump gpump_f ;

class mrp
  {
public:
  struct ibv_mr *_mrp ;
  mrp() :
    _mrp(NULL)
    { }
  mrp(ibv_mr *a) :
    _mrp(a)
    { }
  };

static vector<mrp> mrs ;

void gpump_f_init(MPI_Fint vcomm)
  {
    int rc ;
    rc = gpump_f.init(MPI_Comm_f2c(vcomm)) ;
    if(rc) {
      abort() ;
    }
  }
void gpump_f_register_region(MPI_Fint *mr_index, void * vaddr, gpump_size_t vsize)
  {
//    cerr << "gpump_f_register_region(" << (void *) mrp << "," << vaddr << "," << vsize << endl ;
    struct ibv_mr *mr = gpump_f.register_region(vaddr, vsize) ;
    mrp q(mr) ;
    mrs.push_back(q) ;
    *mr_index=mrs.size() ;
  }
void gpump_f_replace_region(MPI_Fint vmr_index, void * vaddr, gpump_size_t vsize)
  {
//    cerr << "gpump_f_register_region(" << (void *) mrp << "," << vaddr << "," << vsize << endl ;

    assert(vmr_index > 0 && vmr_index <= mrs.size()) ;
    gpump_f.deregister_region(mrs[vmr_index-1]._mrp) ;
    struct ibv_mr *mr = gpump_f.register_region(vaddr, vsize) ;
    mrs[vmr_index-1]._mrp =  mr ;
  }
void gpump_f_deregister_region(MPI_Fint vmr_index)
  {

    assert(vmr_index > 0 && vmr_index <= mrs.size()) ;
    gpump_f.deregister_region(mrs[vmr_index-1]._mrp) ;
    mrs[vmr_index-1]._mrp = NULL ;
  }
void gpump_f_connect_propose(MPI_Fint vtarget)
  {
    gpump_f.connect_propose(vtarget) ;
  }
void gpump_f_connect_accept(MPI_Fint vtarget)
  {
    gpump_f.connect_accept(vtarget) ;
  }
//void gpump_f_connect(MPI_Fint vtarget)
//  {
//    gpump_f.connect(vtarget) ;
//  }
void gpump_f_disconnect(MPI_Fint vtarget)
  {
    gpump_f.disconnect(vtarget) ;
  }
void gpump_f_create_window_propose(MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {
    gpump_f.create_window_propose(vtarget, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_replace_window_propose(MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {
    gpump_f.replace_window_propose(vtarget, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_window_accept(MPI_Fint vtarget)
  {
    gpump_f.window_accept(vtarget) ;
  }

void gpump_f_create_window_propose_x(MPI_Fint vtarget, MPI_Fint *wx, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {
    gpump_f.create_window_propose_x(vtarget, wx, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_replace_window_propose_x(MPI_Fint vtarget, MPI_Fint wx, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {
    gpump_f.replace_window_propose_x(vtarget, wx, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_window_accept_x(MPI_Fint vtarget, MPI_Fint vwx)
  {
    gpump_f.window_accept_x(vtarget, vwx) ;
  }

void gpump_f_create_window(MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {
    gpump_f.create_window(vtarget, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_destroy_window(MPI_Fint vtarget)
  {
    gpump_f.destroy_window(vtarget) ;
  }
void gpump_f_destroy_window_x(MPI_Fint vtarget, MPI_Fint vwx)
  {
    gpump_f.destroy_window_x(vtarget, vwx) ;
  }
void gpump_f_cork(void)
  {
    gpump_f.cork() ;
  }
void gpump_f_uncork(cudaStream_t vstream)
  {
    gpump_f.uncork(vstream) ;
  }
void gpump_f_stream_put(MPI_Fint vtarget, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {
    gpump_f.stream_put(vtarget, vstream,voffset, vremote_offset, vsize) ;
  }
void gpump_f_iput(MPI_Fint vtarget, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {
    gpump_f.iput(vtarget, voffset, vremote_offset, vsize) ;
  }
void gpump_f_stream_wait_put_complete(MPI_Fint vtarget, cudaStream_t vstream)
  {
    gpump_f.stream_wait_put_complete(vtarget, vstream) ;
  }
void gpump_f_cpu_ack_iput(MPI_Fint vtarget)
  {
    gpump_f.cpu_ack_iput(vtarget) ;
  }
void gpump_f_is_put_complete(MPI_Fint vtarget, MPI_Fint *is_complete)
  {
    *is_complete=gpump_f.is_put_complete(vtarget) ;
  }
void gpump_f_wait_put_complete(MPI_Fint vtarget)
  {
    gpump_f.wait_put_complete(vtarget) ;
  }
void gpump_f_stream_get(MPI_Fint vtarget, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {
    gpump_f.stream_get(vtarget, vstream, voffset, vremote_offset, vsize) ;
  }
void gpump_f_iget(MPI_Fint vtarget, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {
    gpump_f.iget(vtarget, voffset, vremote_offset, vsize) ;
  }
void gpump_f_stream_wait_get_complete(MPI_Fint vtarget, cudaStream_t vstream)
  {
    gpump_f.stream_wait_get_complete(vtarget, vstream) ;
  }
void gpump_f_cpu_ack_iget(MPI_Fint vtarget)
  {
    gpump_f.cpu_ack_iget(vtarget) ;
  }
void gpump_f_is_get_complete(MPI_Fint vtarget, MPI_Fint *is_complete)
  {
    *is_complete=gpump_f.is_get_complete(vtarget) ;
  }
void gpump_f_wait_get_complete(MPI_Fint vtarget)
  {
    gpump_f.wait_get_complete(vtarget) ;
  }
void gpump_f_stream_put_x(MPI_Fint vtarget, MPI_Fint vwx, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {
    gpump_f.stream_put_x(vtarget, vwx, vstream,voffset, vremote_offset, vsize) ;
  }
void gpump_f_iput_x(MPI_Fint vtarget, MPI_Fint vwx, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {
    gpump_f.iput_x(vtarget, vwx, voffset, vremote_offset, vsize) ;
  }
void gpump_f_stream_wait_put_complete_x(MPI_Fint vtarget, MPI_Fint vwx, cudaStream_t vstream)
  {
    gpump_f.stream_wait_put_complete_x(vtarget, vwx, vstream) ;
  }
void gpump_f_cpu_ack_iput_x(MPI_Fint vtarget, MPI_Fint vwx)
  {
    gpump_f.cpu_ack_iput_x(vtarget, vwx) ;
  }
void gpump_f_is_put_complete_x(MPI_Fint vtarget, MPI_Fint vwx, MPI_Fint *is_complete)
  {
    *is_complete=gpump_f.is_put_complete_x(vtarget, vwx) ;
  }
void gpump_f_wait_put_complete_x(MPI_Fint vtarget, MPI_Fint vwx)
  {
    gpump_f.wait_put_complete_x(vtarget, vwx) ;
  }
void gpump_f_stream_get_x(MPI_Fint vtarget, MPI_Fint vwx, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {
    gpump_f.stream_get_x(vtarget, vwx, vstream, voffset, vremote_offset, vsize) ;
  }
void gpump_f_iget_x(MPI_Fint vtarget, MPI_Fint vwx, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {
    gpump_f.iget_x(vtarget, vwx, voffset, vremote_offset, vsize) ;
  }
void gpump_f_stream_wait_get_complete_x(MPI_Fint vtarget, MPI_Fint vwx, cudaStream_t vstream)
  {
    gpump_f.stream_wait_get_complete_x(vtarget, vwx, vstream) ;
  }
void gpump_f_cpu_ack_iget_x(MPI_Fint vtarget, MPI_Fint vwx)
  {
    gpump_f.cpu_ack_iget_x(vtarget, vwx) ;
  }
void gpump_f_is_get_complete_x(MPI_Fint vtarget, MPI_Fint vwx, MPI_Fint *is_complete)
  {
    *is_complete=gpump_f.is_get_complete_x(vtarget, vwx) ;
  }
void gpump_f_wait_get_complete_x(MPI_Fint vtarget, MPI_Fint vwx)
  {
    gpump_f.wait_get_complete_x(vtarget, vwx) ;
  }
void gpump_f_stream_send(MPI_Fint vtarget, cudaStream_t vstream, MPI_Fint vsource_mr_index, gpump_size_t voffset, gpump_size_t vsize)
  {
    assert(vsource_mr_index > 0 && vsource_mr_index <= mrs.size()) ;
    gpump_f.stream_send(vtarget, vstream, mrs[(vsource_mr_index)-1]._mrp, voffset, vsize) ;
  }
void gpump_f_isend(MPI_Fint vtarget, MPI_Fint vsource_mr_index, gpump_size_t voffset, gpump_size_t vsize)
  {
    assert(vsource_mr_index > 0 && vsource_mr_index <= mrs.size()) ;
    gpump_f.isend(vtarget, mrs[(vsource_mr_index)-1]._mrp, voffset, vsize) ;
  }
void gpump_f_stream_wait_send_complete(MPI_Fint vtarget, cudaStream_t vstream)
  {
    gpump_f.stream_wait_send_complete(vtarget, vstream) ;
  }
void gpump_f_cpu_ack_isend(MPI_Fint vtarget)
  {
    gpump_f.cpu_ack_isend(vtarget) ;
  }
void gpump_f_is_send_complete(MPI_Fint vtarget, MPI_Fint *is_complete)
  {
    *is_complete=gpump_f.is_send_complete(vtarget) ;
  }
void gpump_f_wait_send_complete(MPI_Fint vtarget)
  {
    gpump_f.wait_send_complete(vtarget) ;
  }
void gpump_f_receive(MPI_Fint vsource, MPI_Fint vtarget_mr_index, gpump_size_t voffset, gpump_size_t vsize)
  {
    assert(vtarget_mr_index > 0 && vtarget_mr_index <= mrs.size()) ;
    gpump_f.receive(vsource, mrs[vtarget_mr_index-1]._mrp, voffset, vsize) ;
  }
void gpump_f_stream_wait_recv_complete(MPI_Fint vsource, cudaStream_t vstream)
  {
    gpump_f.stream_wait_recv_complete(vsource, vstream) ;
  }
void gpump_f_cpu_ack_recv(MPI_Fint vsource)
  {
    gpump_f.cpu_ack_recv(vsource) ;
  }
void gpump_f_is_receive_complete(MPI_Fint vtarget, MPI_Fint *is_complete)
  {
    *is_complete=gpump_f.is_receive_complete(vtarget) ;
  }
void gpump_f_wait_receive_complete(MPI_Fint vsource)
  {
    gpump_f.wait_receive_complete(vsource) ;
  }
void gpump_f_term(void)
  {
    mrs.clear() ;
    gpump_f.term() ;
  }

class gpump_f_r
  {
public:
  gpump _gpump ;
  vector<mrp> _mrs ;
  };

class gpump_f_r_item
  {
public:
    gpump_f_r *_p ;
    gpump_f_r_item() :
      _p(NULL)
      { }
  };

class gpump_f_r_array
  {
  enum {
    k_BaseCount=10
  };
  gpump_f_r *_base[k_BaseCount] ;
  unordered_map<MPI_Fint,gpump_f_r_item> _map ;
public:
  gpump_f_r * get_p(MPI_Fint vcomm)
    {
      unsigned int vcommu=vcomm ;
      return (vcommu < k_BaseCount) ? _base[vcommu] : _map[vcommu]._p ;
    }
  void set_p(MPI_Fint vcomm, gpump_f_r * p)
    {
      unsigned int vcommu=vcomm ;
      if ( vcommu < k_BaseCount)
        {
          _base[vcommu] = p ;
        }
      else
        {
          _map[vcommu]._p = p ;
        }
    }
    void clear_p(MPI_Fint vcomm)
      {
        unsigned int vcommu=vcomm ;
        if ( vcommu < k_BaseCount)
          {
            _base[vcommu] = NULL ;
          }
        else
          {
            _map.erase(vcommu) ;
          }
      }

  };

static gpump_f_r_array gpump_f_r_array ;
//static unordered_map<MPI_Fint,gpump_f_r_item> gpump_f_r_map ;
//
//static gpump_f_r * get_p(MPI_Fint vcomm)
//  {
//     return  gpump_f_r_map[vcomm]._p ;
//  }

void gpump_f_init_r(MPI_Fint vcomm)
  {
    int rc ;

    gpump_f_r * p = new gpump_f_r ;
    gpump_f_r_array.set_p(vcomm, p) ;
    rc = p->_gpump.init(MPI_Comm_f2c(vcomm)) ;
    if(rc) {
      abort() ;
    }

  }

void gpump_f_term_r(MPI_Fint vcomm)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_mrs.clear() ;
    p->_gpump.term() ;
    gpump_f_r_array.clear_p(vcomm) ;
  }

void gpump_f_register_region_r(MPI_Fint vcomm, MPI_Fint *mr_index, void * vaddr, gpump_size_t vsize)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
//    cerr << "gpump_f_register_region(" << (void *) mrp << "," << vaddr << "," << vsize << endl ;
    struct ibv_mr *mr = p->_gpump.register_region(vaddr, vsize) ;
    mrp q(mr) ;
    p->_mrs.push_back(q) ;
    *mr_index = p->_mrs.size() ;
  }
void gpump_f_replace_region_r(MPI_Fint vcomm, MPI_Fint vmr_index, void * vaddr, gpump_size_t vsize)
  {


    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    assert(vmr_index > 0 && vmr_index <= p->_mrs.size()) ;
    p->_gpump.deregister_region(p->_mrs[(vmr_index)-1]._mrp) ;
//    cerr << "gpump_f_replace_region(" << (void *) mrp << "," << vaddr << "," << vsize << endl ;
    struct ibv_mr *mr = p->_gpump.register_region(vaddr, vsize) ;
    p->_mrs[vmr_index-1]._mrp = mr ;
  }
void gpump_f_deregister_region_r(MPI_Fint vcomm, MPI_Fint vmr_index)
  {


    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    assert(vmr_index > 0 && vmr_index <= p->_mrs.size()) ;
    p->_gpump.deregister_region(p->_mrs[vmr_index-1]._mrp) ;
    p->_mrs[vmr_index-1]._mrp = NULL ;
  }
void gpump_f_connect_propose_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.connect_propose(vtarget) ;
  }
void gpump_f_connect_accept_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.connect_accept(vtarget) ;
  }
//void gpump_f_connect_r(MPI_Fint vcomm, MPI_Fint vtarget)
//  {
//    p->_gpump.connect(vtarget) ;
//  }
void gpump_f_disconnect_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.disconnect(vtarget) ;
  }
void gpump_f_create_window_propose_r(MPI_Fint vcomm, MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.create_window_propose(vtarget, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_replace_window_propose_r(MPI_Fint vcomm, MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.replace_window_propose(vtarget, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_window_accept_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.window_accept(vtarget) ;
  }
void gpump_f_create_window_propose_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint *index, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    int x ;
    p->_gpump.create_window_propose_x(vtarget, &x, vlocal_address, vremote_address, vsize) ;
    *index = x+1 ;
  }
void gpump_f_replace_window_propose_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {


    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.replace_window_propose_x(vtarget, vindex-1, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_window_accept_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.window_accept_x(vtarget,vindex-1) ;
  }
void gpump_f_create_window_r(MPI_Fint vcomm, MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.create_window(vtarget, vlocal_address, vremote_address, vsize) ;
  }
void gpump_f_destroy_window_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.destroy_window(vtarget) ;
  }
void gpump_f_destroy_window_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.destroy_window_x(vtarget, vindex-1) ;
  }
void gpump_f_cork_r(MPI_Fint vcomm)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.cork() ;
  }
void gpump_f_uncork_r(MPI_Fint vcomm, cudaStream_t vstream)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.uncork(vstream) ;
  }
void gpump_f_stream_put_r(MPI_Fint vcomm, MPI_Fint vtarget, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_put(vtarget, vstream,voffset, vremote_offset, vsize) ;
  }
void gpump_f_iput_r(MPI_Fint vcomm, MPI_Fint vtarget, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.iput(vtarget, voffset, vremote_offset, vsize) ;
  }
void gpump_f_stream_wait_put_complete_r(MPI_Fint vcomm, MPI_Fint vtarget, cudaStream_t vstream)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_wait_put_complete(vtarget, vstream) ;
  }
void gpump_f_cpu_ack_iput_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.cpu_ack_iput(vtarget) ;
  }
void gpump_f_is_put_complete_r(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint *is_complete)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    *is_complete=p->_gpump.is_put_complete(vtarget) ;
  }
void gpump_f_wait_put_complete_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.wait_put_complete(vtarget) ;
  }
void gpump_f_stream_get_r(MPI_Fint vcomm, MPI_Fint vtarget, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_get(vtarget, vstream, voffset, vremote_offset, vsize) ;
  }
void gpump_f_iget_r(MPI_Fint vcomm, MPI_Fint vtarget, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.iget(vtarget, voffset, vremote_offset, vsize) ;
  }
void gpump_f_stream_wait_get_complete_r(MPI_Fint vcomm, MPI_Fint vtarget, cudaStream_t vstream)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_wait_get_complete(vtarget, vstream) ;
  }
void gpump_f_cpu_ack_iget_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.cpu_ack_iget(vtarget) ;
  }
void gpump_f_is_get_complete_r(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint *is_complete)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    *is_complete=p->_gpump.is_get_complete(vtarget) ;
  }
void gpump_f_wait_get_complete_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.wait_get_complete(vtarget) ;
  }
void gpump_f_stream_put_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_put_x(vtarget, vindex-1, vstream, voffset, vremote_offset, vsize) ;
  }
void gpump_f_iput_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.iput_x(vtarget, vindex-1, voffset, vremote_offset, vsize) ;
  }
void gpump_f_stream_wait_put_complete_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, cudaStream_t vstream)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_wait_put_complete_x(vtarget, vindex-1, vstream) ;
  }
void gpump_f_cpu_ack_iput_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.cpu_ack_iput_x(vtarget, vindex-1) ;
  }
void gpump_f_is_put_complete_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, MPI_Fint *is_complete)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    *is_complete=p->_gpump.is_put_complete_x(vtarget, vindex-1) ;
  }
void gpump_f_wait_put_complete_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.wait_put_complete_x(vtarget, vindex-1) ;
  }
void gpump_f_stream_get_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_get_x(vtarget, vindex-1, vstream, voffset, vremote_offset, vsize) ;
  }
void gpump_f_iget_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize )
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.iget_x(vtarget, vindex-1, voffset, vremote_offset, vsize) ;
  }
void gpump_f_stream_wait_get_complete_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, cudaStream_t vstream)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_wait_get_complete_x(vtarget, vindex-1, vstream) ;
  }
void gpump_f_cpu_ack_iget_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.cpu_ack_iget_x(vtarget, vindex-1) ;
  }
void gpump_f_is_get_complete_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex, MPI_Fint *is_complete)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    *is_complete=p->_gpump.is_get_complete_x(vtarget, vindex-1) ;
  }
void gpump_f_wait_get_complete_rx(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vindex)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.wait_get_complete_x(vtarget, vindex-1) ;
  }
void gpump_f_stream_send_r(MPI_Fint vcomm, MPI_Fint vtarget, cudaStream_t vstream, MPI_Fint vsource_mr_index, gpump_size_t voffset, gpump_size_t vsize)
  {
    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    assert(vsource_mr_index > 0 && vsource_mr_index <= p->_mrs.size()) ;
    p->_gpump.stream_send(vtarget, vstream, p->_mrs[vsource_mr_index-1]._mrp, voffset, vsize) ;
  }
void gpump_f_isend_r(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vsource_mr_index, gpump_size_t voffset, gpump_size_t vsize)
  {
    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    assert(vsource_mr_index > 0 && vsource_mr_index <= p->_mrs.size()) ;
    p->_gpump.isend(vtarget, p->_mrs[vsource_mr_index-1]._mrp, voffset, vsize) ;
  }
void gpump_f_stream_wait_send_complete_r(MPI_Fint vcomm, MPI_Fint vtarget, cudaStream_t vstream)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_wait_send_complete(vtarget, vstream) ;
  }
void gpump_f_cpu_ack_isend_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.cpu_ack_isend(vtarget) ;
  }
void gpump_f_is_send_complete_r(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint *is_complete)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    *is_complete=p->_gpump.is_send_complete(vtarget) ;
  }
void gpump_f_wait_send_complete_r(MPI_Fint vcomm, MPI_Fint vtarget)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ; assert(p != NULL) ;
    p->_gpump.wait_send_complete(vtarget) ;
  }
void gpump_f_receive_r(MPI_Fint vcomm, MPI_Fint vsource, MPI_Fint vtarget_mr_index, gpump_size_t voffset, gpump_size_t vsize)
  {
    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    assert(vtarget_mr_index > 0 && vtarget_mr_index <= p->_mrs.size()) ;
    p->_gpump.receive(vsource, p->_mrs[vtarget_mr_index-1]._mrp, voffset, vsize) ;
  }
void gpump_f_stream_wait_recv_complete_r(MPI_Fint vcomm, MPI_Fint vsource, cudaStream_t vstream)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.stream_wait_recv_complete(vsource, vstream) ;
  }
void gpump_f_cpu_ack_recv_r(MPI_Fint vcomm, MPI_Fint vsource)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.cpu_ack_recv(vsource) ;
  }
void gpump_f_is_receive_complete_r(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint *is_complete)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    *is_complete=p->_gpump.is_receive_complete(vtarget) ;
  }
void gpump_f_wait_receive_complete_r(MPI_Fint vcomm, MPI_Fint vsource)
  {

    gpump_f_r * p = gpump_f_r_array.get_p(vcomm) ;
    assert(p != NULL) ;
    p->_gpump.wait_receive_complete(vsource) ;
  }
