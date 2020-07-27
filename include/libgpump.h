// Copyright (C) IBM Corporation 2018. All Rights Reserved
//
//    This program is licensed under the terms of the Eclipse Public License
//    v1.0 as published by the Eclipse Foundation and available at
//    http://www.eclipse.org/legal/epl-v10.html
//
//    
//    
// $COPYRIGHT$

#ifndef __libgpump_h__
#define __libgpump_h__

#include <mpi.h>
#include <infiniband/verbs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>

struct gpump ;
#ifdef __cplusplus
extern "C" {
#endif

typedef size_t gpump_size_t ;

// C-style interface
  /**
   * \brief Initialize the gpump library
   *
   * \param [in] comm  MPI communicator
   * \retval opaque pointer to gpump object
   *
   */
struct gpump * gpump_init(MPI_Comm comm) ;
  /**
   * \brief Terminate use of the gpump library
   *
   * \param [in] g opaque pointer to gpump object
   *
   */
void gpump_term(struct gpump * g ) ;
  /**
   * \brief Register a region of GPU memory for use with the gpump library
   *
   * \param [in] g opaque pointer to gpump object
   * \param [in] addr address as returned by cudaMalloc
   * \param [in] size size in bytes of memory region
   *
   * \retval address of resulting struct ibv_mr describing the memory region
   *
   */
struct ibv_mr * gpump_register_region(struct gpump * g , void * addr, size_t size) ;
  /**
   * \brief Deregister a region of GPU memory
   *
   * \param [in] g opaque pointer to gpump object
   * \param [in] mrp address of struct ibv_mr as returned by the gpump_register_reguion call
   *
   */
void gpump_deregister_region(struct gpump * g , struct ibv_mr * mrp) ;
  /**
   * \brief Propose connecting 2 ranks for gpump data exchange
   *
   * \note This function needs to be called by both communication partners.
   *
   * \param [in] g opaque pointer to gpump object
   * \param [in] target rank in communicator of communication partner
   *
   */
void gpump_connect_propose(struct gpump * g , int target) ;
/**
 * \brief Accept proposal to connect 2 ranks for gpump data exchange
 *
 * \note This function needs to be called by both communication partners. It is a blocking call.
 *
 * \param [in] g opaque pointer to gpump object
 * \param [in] target rank in communicator of communication partner
 *
 */
void gpump_connect_accept(struct gpump * g , int target) ;
///**
// * \brief Connect 2 ranks for gpump data exchange
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param [in] g opaque pointer to gpump object
// * \param [in] target rank in communicator of communication partner
// *
// */
//void gpump_connect(struct gpump * g , int target) ;
  /**
   * \brief Disconnect 2 ranks that have been connected for gpump data exchange
   *
   * \param [in] g opaque pointer to gpump object
   * \param [in] target rank in communicator of communication partner
   *
   */
void gpump_disconnect(struct gpump * g, int target) ;
  /**
   * \brief Propose creation of a pair of windows for RDMA put and get operations
   *
   * \note This function needs to be called by both communication partners.
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
   * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
   * \param[in] size size in bytes of local and remote windows
   *
   */
void gpump_create_window_propose(struct gpump * g , int target, void * local_address, void * remote_address, size_t size) ;
/**
 * \brief Propose creation of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_replace_window_propose(struct gpump * g , int target, void * local_address, void * remote_address, size_t size) ;
/**
 * \brief Accept a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners. It is a blocking call
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 *
 */
void gpump_window_accept(struct gpump * g , int target) ;
///**
// * \brief Create a pair of windows for RDMA put and get operations
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param[in] g opaque pointer to gpump object
// * \param[in] target rank in communicator of communication partner
// * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
// * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
// * \param[in] size size in bytes of local and remote windows
// *
// */
//void gpump_create_window(struct gpump * g , int target, void * local_address, void * remote_address, size_t size) ;
  /**
   * \brief Destroy the RDMA put/get window
   *
   * \note This function is usually used before establishing a new pair of windows with another gpump_create_window call.
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   */
void gpump_destroy_window(struct gpump * g , int target) ;
/**
 * \brief Propose creation of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[out] wx index in window array
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_create_window_propose_x(struct gpump * g , int target, int *wx, void * local_address, void * remote_address, size_t size) ;
/**
 * \brief Propose replacement of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_replace_window_propose_x(struct gpump * g , int target, int wx, void * local_address, void * remote_address, size_t size) ;
/**
* \brief Accept a pair of windows for RDMA put and get operations
*
* \note This function needs to be called by both communication partners. It is a blocking call
*
* \param[in] g opaque pointer to gpump object
* \param[in] target rank in communicator of communication partner
* \param[in] wx index in window array
*
*/
void gpump_window_accept_x(struct gpump * g , int target, int wx) ;
///**
// * \brief Create a pair of windows for RDMA put and get operations
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param[in] g opaque pointer to gpump object
// * \param[in] target rank in communicator of communication partner
// * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
// * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
// * \param[in] size size in bytes of local and remote windows
// *
// */
//void gpump_create_window(struct gpump * g , int target, void * local_address, void * remote_address, size_t size) ;
/**
 * \brief Destroy the RDMA put/get window
 *
 * \note This function is usually used before establishing a new pair of windows with another gpump_create_window call.
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 */
void gpump_destroy_window(struct gpump * g , int target) ;
/**
 * \brief Destroy the RDMA put/get window
 *
 * \note This function is usually used before establishing a new pair of windows with another gpump_create_window call.
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 */
void gpump_destroy_window_x(struct gpump * g , int target, int wx) ;
/**
 * \brief Cork the put/get/send calls to the GPU. Put/get/send requests are deferred (batched) and sent to the GPU
 * on the uncork call
 *
 * \param[in] g opaque pointer to gpump object
 */
void gpump_cork(struct gpump * g) ;
/**
 * \brief Uncork put/get/send calls to the GPU
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] stream CUDA stream which will initiate the requests to the HCA
 */
void gpump_uncork(struct gpump * g, cudaStream_t stream) ;
  /**
   * \brief Nonblocking RDMA put on stream to communication partner
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   * \param[in] stream CUDA stream which will initiate the put request to the HCA
   * \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
   * \param[out] remote_offset Offset in bytes from the communication partner's ' remote address' in the window which data will be written to
   * \param[in] size Size in bytes of data transfer
   */
void gpump_stream_put(struct gpump * g , int target, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size ) ;
  /**
   * \brief Nonblocking RDMA put triggered by CPU to communication partner
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   * \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
   * \param[out] remote_offset Offset in bytes from the communication partner's 'remote address' in the window which data will be written to
   * \param[in] size Size in bytes of data transfer
   */
void gpump_iput(struct gpump * g , int target, size_t offset, size_t remote_offset, size_t size ) ;
  /**
   * \brief Block the stream until the RDMA put completes
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   * \param[in] stream CUDA stream to block until the RDMA put completes
   *
   */
void gpump_stream_wait_put_complete(struct gpump * g , int target, cudaStream_t stream) ;
  /**
   * \brief CPU acknowledges that RDMA put will complete
   *
   * \note Make this call if you will not be making a call to gpump_stream_wait_put_complete for this transfer
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   */
void gpump_cpu_ack_iput(struct gpump *g, int target) ;
  /**
   * \brief Nonblocking test of whether the RDMA put is complete
   *
   * \note '1' will be returned once only for each RDMA put operation
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   *
   * \retval 0 The RDMA is still in progress
   * \retval 1 The RDMA is complete
   */
int gpump_is_put_complete(struct gpump *g,int target) ;
  /**
   * \brief Blocking wait for RDMA completion
   *
   * \note equivalent to calling gpump_is_put_complete in a loop until 1 is returned
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   *
   */
void gpump_wait_put_complete(struct gpump * g , int target) ;
  /**
   * \brief Nonblocking RDMA get on stream from communication partner
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   * \param[in] stream CUDA stream which will initiate the get request to the HCA
   * \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
   * \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
   * \param[in] size Size in bytes of data transfer
   */
void gpump_stream_get(struct gpump * g , int target, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size ) ;
/**
 * \brief Nonblocking RDMA get triggered by CPU from communication partner
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
 * \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_iget(struct gpump * g , int target, size_t offset, size_t remote_offset, size_t size ) ;
/**
 * \brief Block the stream until the RDMA get completes
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] stream CUDA stream to block until the RDMA get completes
 *
 */
void gpump_stream_wait_get_complete(struct gpump * g , int target, cudaStream_t stream) ;
/**
 * \brief CPU acknowledges that RDMA get will complete
 *
 * \note Make this call if you will not be making a call to gpump_stream_wait_get_complete for this transfer
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 */
void gpump_cpu_ack_iget(struct gpump *g, int target) ;
/**
 * \brief Nonblocking test of whether the RDMA get is complete
 *
 * \note '1' will be returned once only for each RDMA get operation
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 *
 * \retval 0 The RDMA is still in progress
 * \retval 1 The RDMA is complete
 */
int gpump_is_get_complete(struct gpump *g,int target) ;
  /**
   * \brief Blocking wait for RDMA completion
   *
   * \note equivalent to calling gpump_is_get_complete in a loop until 1 is returned
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   *
   */
void gpump_wait_get_complete(struct gpump * g , int target) ;
  /**
   * \brief Nonblocking send for 2-sided communication on stream to communication partner
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   * \param[in] stream CUDA stream which will initiate the send request to the HCA
   * \param[in] source_mr memory region as returned by gpump_register_region from which data will be taken
   * \param[in] offset Offset in bytes from in the source_mr from which data will be taken
   * \param[in] size Size in bytes of data transfer
   */
/**
 * \brief Nonblocking RDMA put on stream to communication partner
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] stream CUDA stream which will initiate the put request to the HCA
 * \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
 * \param[out] remote_offset Offset in bytes from the communication partner's ' remote address' in the window which data will be written to
 * \param[in] size Size in bytes of data transfer
 */
void gpump_stream_put_x(struct gpump * g , int target, int wx, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size ) ;
/**
 * \brief Nonblocking RDMA put triggered by CPU to communication partner
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
 * \param[out] remote_offset Offset in bytes from the communication partner's 'remote address' in the window which data will be written to
 * \param[in] size Size in bytes of data transfer
 */
void gpump_iput_x(struct gpump * g , int target, int wx, size_t offset, size_t remote_offset, size_t size ) ;
/**
 * \brief Block the stream until the RDMA put completes
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] stream CUDA stream to block until the RDMA put completes
 *
 */
void gpump_stream_wait_put_complete_x(struct gpump * g , int target, int wx, cudaStream_t stream) ;
/**
 * \brief CPU acknowledges that RDMA put will complete
 *
 * \note Make this call if you will not be making a call to gpump_stream_wait_put_complete for this transfer
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 */
void gpump_cpu_ack_iput_x(struct gpump *g, int target, int wx) ;
/**
 * \brief Nonblocking test of whether the RDMA put is complete
 *
 * \note '1' will be returned once only for each RDMA put operation
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 *
 * \retval 0 The RDMA is still in progress
 * \retval 1 The RDMA is complete
 */
int gpump_is_put_complete_x(struct gpump *g,int target, int wx) ;
/**
 * \brief Blocking wait for RDMA completion
 *
 * \note equivalent to calling gpump_is_put_complete in a loop until 1 is returned
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] wx index in window array
 * \param[in] target rank in communicator of communication partner
 *
 */
void gpump_wait_put_complete_x(struct gpump * g , int target, int wx) ;
/**
 * \brief Nonblocking RDMA get on stream from communication partner
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] stream CUDA stream which will initiate the get request to the HCA
 * \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
 * \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_stream_get_x(struct gpump * g , int target, int wx, cudaStream_t stream, size_t offset, size_t remote_offset, size_t size ) ;
/**
* \brief Nonblocking RDMA get triggered by CPU from communication partner
*
* \param[in] g opaque pointer to gpump object
* \param[in] target rank in communicator of communication partner
* \param[in] wx index in window array
* \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
* \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
* \param[in] size Size in bytes of data transfer
*/
void gpump_iget_x(struct gpump * g , int target, int wx, size_t offset, size_t remote_offset, size_t size ) ;
/**
* \brief Block the stream until the RDMA get completes
*
* \param[in] g opaque pointer to gpump object
* \param[in] target rank in communicator of communication partner
* \param[in] wx index in window array
* \param[in] stream CUDA stream to block until the RDMA get completes
*
*/
void gpump_stream_wait_get_complete_x(struct gpump * g , int target, int wx, cudaStream_t stream) ;
/**
* \brief CPU acknowledges that RDMA get will complete
*
* \note Make this call if you will not be making a call to gpump_stream_wait_get_complete for this transfer
*
* \param[in] g opaque pointer to gpump object
* \param[in] target rank in communicator of communication partner
* \param[in] wx index in window array
*/
void gpump_cpu_ack_iget_x(struct gpump *g, int target, int wx) ;
/**
* \brief Nonblocking test of whether the RDMA get is complete
*
* \note '1' will be returned once only for each RDMA get operation
*
* \param[in] g opaque pointer to gpump object
* \param[in] target rank in communicator of communication partner
* \param[in] wx index in window array
*
* \retval 0 The RDMA is still in progress
* \retval 1 The RDMA is complete
*/
int gpump_is_get_complete_x(struct gpump *g,int target, int is_complete) ;
/**
 * \brief Blocking wait for RDMA completion
 *
 * \note equivalent to calling gpump_is_get_complete in a loop until 1 is returned
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 *
 */
void gpump_wait_get_complete_x(struct gpump * g , int target, int wx) ;
/**
 * \brief Nonblocking send for 2-sided communication on stream to communication partner
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 * \param[in] stream CUDA stream which will initiate the send request to the HCA
 * \param[in] source_mr memory region as returned by gpump_register_region from which data will be taken
 * \param[in] offset Offset in bytes from in the source_mr from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_stream_send(struct gpump * g , int target, cudaStream_t stream, struct ibv_mr * source_mr, size_t offset, size_t size) ;
  /**
   * \brief Nonblocking send for 2-sided communication triggered by CPU to communication partner
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   * \param[in] source_mr memory region as returned by gpump_register_region from which data will be taken
   * \param[in] offset Offset in bytes from in the source_mr from which data will be taken
   * \param[in] size Size in bytes of data transfer
   */
void gpump_isend(struct gpump * g , int target, struct ibv_mr * source_mr, size_t offset, size_t size) ;
  /**
   * \brief Block the stream until the send completes
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   * \param[in] stream CUDA stream to block until the send completes
   *
   */
void gpump_stream_wait_send_complete(struct gpump * g , int target, cudaStream_t stream) ;
  /**
   * \brief CPU acknowledges that send will complete
   *
   * \note Make this call if you will not be making a call to gpump_stream_wait_send_complete for this transfer
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   */
void gpump_cpu_ack_isend(struct gpump *g, int target) ;
  /**
   * \brief Nonblocking test of whether the send is complete
   *
   * \note '1' will be returned once only for each send operation
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   *
   * \retval 0 The send is still in progress
   * \retval 1 The send is complete
   */
int gpump_is_send_complete(struct gpump *g,int target) ;
  /**
   * \brief Blocking wait for send completion
   *
   * \note equivalent to calling gpump_is_send_complete in a loop until 1 is returned
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] target rank in communicator of communication partner
   *
   */
void gpump_wait_send_complete(struct gpump * g , int target) ;
  /**
   * \brief Nonblocking receive for 2-sided communication triggered by CPU to communication partner
   *
   * \note The underlying hardware does not support a 'gpump_stream_receive'
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] source rank in communicator of communication partner
   * \param[out] target_mr memory region as returned by gpump_register_region to which data will be written
   * \param[in] offset Offset in bytes from in the target_mr to which data will be written
   * \param[in] size Size in bytes of data transfer
   */
void gpump_receive(struct gpump * g , int source, struct ibv_mr * target_mr, size_t offset, size_t size) ;
  /**
   * \brief Block the stream until the receive completes
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] source rank in communicator of communication partner
   * \param[in] stream CUDA stream to block until the receive completes
   *
   */
void gpump_stream_wait_recv_complete(struct gpump * g , int source, cudaStream_t stream) ;
  /**
   * \brief CPU acknowledges that receive will complete
   *
   * \note Make this call if you will not be making a call to gpump_stream_wait_recv_complete for this transfer
   *
   * \param[in] g opaque pointer to gpump object
   * \param[in] source rank in communicator of communication partner
   */
void gpump_cpu_ack_recv(struct gpump *g, int source) ;
/**
 * \brief Nonblocking test of whether the receive is complete
 *
 * \note '1' will be returned once only for each receive operation
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] target rank in communicator of communication partner
 *
 * \retval 0 The send is still in progress
 * \retval 1 The send is complete
 */
int gpump_is_receive_complete(struct gpump *g,int target) ;
/**
 * \brief Blocking wait for receive completion
 *
 * \note equivalent to calling gpump_is_receive_complete in a loop until 1 is returned
 *
 * \param[in] g opaque pointer to gpump object
 * \param[in] source rank in communicator of communication partner
 *
 */
void gpump_wait_receive_complete(struct gpump * g , int source) ;

// Fortran-style interface
  /**
   * \brief Initialize the gpump library
   *
   * \note The Fortran interface is modelled after the C interface, with differences
   *  1) The Fortran interface uses a static 'struct gpump', so do not try having 2 gpump_f_init calls active.
   *  2) C pointers for memory regions are changed to array indices. The gpump library maintains the array.
   *  3) GPU addresses are to be passed by writing C_PTR(LOC(x)) in Fortran
   *
   * \param[in] comm The NPI communicator to use
   */
void gpump_f_init(MPI_Fint vcomm) ;
  /**
   * \brief Terminate use of the gpump library
   *
   */
void gpump_f_term(void) ;

  /**
   * \brief Register a region of GPU memory for use with the gpump library
   *
   * \param [out] mr_index Index to the memory region array for this region. Indices start at 1 to match the Fortran convention.
   * \param [in] addr address as returned by cudaMalloc
   * \param [in] size size in bytes of memory region
   *
   * \retval address of resulting struct ibv_mr describing the memory region
   *
   */
void gpump_f_register_region(MPI_Fint *mr_index,  void * vaddr, gpump_size_t vsize) ;
/**
 * \brief Replace registration of a region of GPU memory for use with the gpump library
 *
 * \param [in] mr_index Index to the memory region array for this region. Indices start at 1 to match the Fortran convention.
 * \param [in] addr address as returned by cudaMalloc
 * \param [in] size size in bytes of memory region
 *
 * \retval address of resulting struct ibv_mr describing the memory region
 *
 */
void gpump_f_replace_region(MPI_Fint vmr_index,  void * vaddr, gpump_size_t vsize) ;
  /**
   * \brief Deregister a region of GPU memory
   *
   * \param [in] mr_index Index to the memory region array for this region. Indices start at 1 to match the Fortran convention.
   *
   */
void gpump_f_deregister_region(MPI_Fint vmr_index) ;
/**
 * \brief Set the window count between 2 ranks
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param [in] target rank in communicator of communication partner
 * \param [in] count number of window pairs for RDMA communication
 */
void gpump_f_set_window_count(MPI_Fint vtarget, MPI_Fint vcount) ;
  /**
   * \brief Propose connection of 2 ranks for gpump data exchange
   *
   * \note This function needs to be called by both communication partners.
   *
   * \param [in] target rank in communicator of communication partner
   *
   */
void gpump_f_connect_propose(MPI_Fint vtarget) ;
/**
 * \brief Accept connection of 2 ranks for gpump data exchange
 *
 * \note This function needs to be called by both communication partners. It is a blocking call.
 *
 * \param [in] target rank in communicator of communication partner
 *
 */
void gpump_f_connect_accept(MPI_Fint vtarget) ;
///**
// * \brief Connect 2 ranks for gpump data exchange
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param [in] target rank in communicator of communication partner
// *
// */
//void gpump_f_connect(MPI_Fint vtarget) ;
/**
 * \brief Disconnect 2 ranks that have been connected for gpump data exchange
 *
 * \param [in] target rank in communicator of communication partner
 *
 */
void gpump_f_disconnect(MPI_Fint vtarget) ;
  /**
   * \brief Propose creation of a pair of windows for RDMA put and get operations
   *
   * \note This function needs to be called by both communication partners.
   *
   * \param[in] target rank in communicator of communication partner
   * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
   * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
   * \param[in] size size in bytes of local and remote windows
   *
   */
void gpump_f_create_window_propose(MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
 * \brief Propose replacement of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_f_replace_window_propose(MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
 * \brief Accept creation of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners. It is a blocking call.
 *
 * \param[in] target rank in communicator of communication partner
 *
 */
void gpump_f_window_accept(MPI_Fint vtarget) ;
///**
// * \brief Create a pair of windows for RDMA put and get operations
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param[in] target rank in communicator of communication partner
// * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
// * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
// * \param[in] size size in bytes of local and remote windows
// *
// */
//void gpump_f_create_window(MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
  /**
   * \brief Destroy the RDMA put/get window
   *
   * \note This function is usually used before establishing a new pair of windows with another gpump_create_window call.
   *
   * \param[in] target rank in communicator of communication partner
   */
void gpump_f_destroy_window(MPI_Fint vtarget) ;
/**
 * \brief Propose creation of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] target rank in communicator of communication partner
 * \param[out] wx index in window array
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_f_create_window_propose_x(MPI_Fint vtarget, MPI_Fint *wx, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
 * \brief Propose replacement of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_f_replace_window_propose_x(MPI_Fint vtarget, MPI_Fint wx, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
* \brief Accept a pair of windows for RDMA put and get operations
*
* \note This function needs to be called by both communication partners. It is a blocking call.
*
* \param[in] target rank in communicator of communication partner
* \param[in] wx index in window array
*
*/
void gpump_f_window_accept_x(MPI_Fint vtarget, MPI_Fint vwx) ;
///**
// * \brief Create a pair of windows for RDMA put and get operations
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param[in] target rank in communicator of communication partner
// * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
// * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
// * \param[in] size size in bytes of local and remote windows
// *
// */
//void gpump_f_create_window(MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
 * \brief Destroy the RDMA put/get window
 *
 * \note This function is usually used before establishing a new pair of windows with another gpump_create_window call.
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 */
void gpump_f_destroy_window_x(MPI_Fint vtarget, MPI_Fint vwx) ;
  /**
   * \brief Cork the put/get/send calls to the GPU. Put/get/send requests are deferred (batched) and sent to the GPU
   * on the uncork call
   */
void gpump_f_cork(void) ;
  /**
   * \brief Uncork put/get/send calls to the GPU
   *
   * \param[in] stream CUDA stream which will initiate the requests to the HCA
   */
void gpump_f_uncork(cudaStream_t vstream) ;
/**
 * \brief Nonblocking RDMA put on stream to communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] stream CUDA stream which will initiate the put request to the HCA
 * \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
 * \param[out] remote_offset Offset in bytes from the communication partner's ' remote address' in the window which data will be written to
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_stream_put(MPI_Fint vtarget, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
 * \brief Nonblocking RDMA put triggered by CPU to communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
 * \param[out] remote_offset Offset in bytes from the communication partner's 'remote address' in the window which data will be written to
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_iput(MPI_Fint vtarget, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
 * \brief Block the stream until the RDMA put completes
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] stream CUDA stream to block until the RDMA put completes
 *
 */
void gpump_f_stream_wait_put_complete(MPI_Fint vtarget, cudaStream_t vstream) ;
/**
 * \brief CPU acknowledges that RDMA put will complete
 *
 * \note Make this call if you will not be making a call to gpump_stream_wait_put_complete for this transfer
 *
 * \param[in] target rank in communicator of communication partner
 */
void gpump_f_cpu_ack_iput(MPI_Fint vtarget) ;
/**
 * \brief Nonblocking test of whether the RDMA put is complete
 *
 * \note '1' will be returned once only for each RDMA put operation
 *
 * \param[in] target rank in communicator of communication partner
 * \param[out] is_complete 0 for still in progress, 1 for complete
 *
 */
void gpump_f_is_put_complete(MPI_Fint vtarget, MPI_Fint *is_complete) ;
/**
 * \brief Blocking wait for RDMA completion
 *
 * \note equivalent to calling gpump_f_is_put_complete in a loop until 1 is returned
 *
 * \param[in] target rank in communicator of communication partner
 *
 */
void gpump_f_wait_put_complete(MPI_Fint vtarget) ;
/**
 * \brief Nonblocking RDMA get on stream from communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] stream CUDA stream which will initiate the get request to the HCA
 * \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
 * \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_stream_get(MPI_Fint vtarget, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
 * \brief Nonblocking RDMA get triggered by CPU from communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
 * \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_iget(MPI_Fint vtarget, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
 * \brief Block the stream until the RDMA get completes
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] stream CUDA stream to block until the RDMA get completes
 *
 */
void gpump_f_stream_wait_get_complete(MPI_Fint vtarget, cudaStream_t vstream) ;
/**
 * \brief CPU acknowledges that RDMA get will complete
 *
 * \note Make this call if you will not be making a call to gpump_stream_wait_get_complete for this transfer
 *
 * \param[in] target rank in communicator of communication partner
 */
void gpump_f_cpu_ack_iget(MPI_Fint vtarget) ;
/**
 * \brief Nonblocking test of whether the RDMA get is complete
 *
 * \note '1' will be returned once only for each RDMA get operation
 *
 * \param[in] target rank in communicator of communication partner
 * \param[out] is_complete 0 for still in progress, 1 for complete
 *
 */
void gpump_f_is_get_complete(MPI_Fint vtarget, MPI_Fint *is_complete) ;
/**
 * \brief Blocking wait for RDMA completion
 *
 * \note equivalent to calling gpump_f_is_get_complete in a loop until 1 is returned
 *
 * \param[in] target rank in communicator of communication partner
 *
 */
void gpump_f_wait_get_complete(MPI_Fint vtarget) ;
/**
 * \brief Nonblocking RDMA put on stream to communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] stream CUDA stream which will initiate the put request to the HCA
 * \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
 * \param[out] remote_offset Offset in bytes from the communication partner's ' remote address' in the window which data will be written to
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_stream_put_x(MPI_Fint vtarget, MPI_Fint vwx, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
 * \brief Nonblocking RDMA put triggered by CPU to communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
 * \param[out] remote_offset Offset in bytes from the communication partner's 'remote address' in the window which data will be written to
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_iput_x(MPI_Fint vtarget, MPI_Fint vwx, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
 * \brief Block the stream until the RDMA put completes
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] stream CUDA stream to block until the RDMA put completes
 *
 */
void gpump_f_stream_wait_put_complete_x(MPI_Fint vtarget, MPI_Fint vwx, cudaStream_t vstream) ;
/**
 * \brief CPU acknowledges that RDMA put will complete
 *
 * \note Make this call if you will not be making a call to gpump_stream_wait_put_complete for this transfer
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 */
void gpump_f_cpu_ack_iput_x(MPI_Fint vtarget, MPI_Fint vwx) ;
/**
 * \brief Nonblocking test of whether the RDMA put is complete
 *
 * \note '1' will be returned once only for each RDMA put operation
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[out] is_complete 0 for still in progress, 1 for complete
 *
 */
void gpump_f_is_put_complete_x(MPI_Fint vtarget, MPI_Fint vwx, MPI_Fint *is_complete) ;
/**
 * \brief Blocking wait for RDMA completion
 *
 * \note equivalent to calling gpump_f_is_put_complete in a loop until 1 is returned
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 *
 */
void gpump_f_wait_put_complete_x(MPI_Fint vtarget, MPI_Fint vwx) ;
/**
 * \brief Nonblocking RDMA get on stream from communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] stream CUDA stream which will initiate the get request to the HCA
 * \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
 * \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_stream_get_x(MPI_Fint vtarget, MPI_Fint vwx, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
 * \brief Nonblocking RDMA get triggered by CPU from communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
 * \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_iget_x(MPI_Fint vtarget, MPI_Fint vwx, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
 * \brief Block the stream until the RDMA get completes
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] stream CUDA stream to block until the RDMA get completes
 *
 */
void gpump_f_stream_wait_get_complete_x(MPI_Fint vtarget, MPI_Fint vwx, cudaStream_t vstream) ;
/**
 * \brief CPU acknowledges that RDMA get will complete
 *
 * \note Make this call if you will not be making a call to gpump_stream_wait_get_complete for this transfer
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 */
void gpump_f_cpu_ack_iget_x(MPI_Fint vtarget, MPI_Fint vwx) ;
/**
 * \brief Nonblocking test of whether the RDMA get is complete
 *
 * \note '1' will be returned once only for each RDMA get operation
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[out] is_complete 0 for still in progress, 1 for complete
 *
 */
void gpump_f_is_get_complete_x(MPI_Fint vtarget, MPI_Fint vwx, MPI_Fint *is_complete) ;
/**
 * \brief Blocking wait for RDMA completion
 *
 * \note equivalent to calling gpump_f_is_get_complete in a loop until 1 is returned
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 *
 */
void gpump_f_wait_get_complete_x(MPI_Fint vtarget, MPI_Fint vwx) ;
/**
 * \brief Nonblocking send for 2-sided communication on stream to communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] wx index in window array
 * \param[in] stream CUDA stream which will initiate the send request to the HCA
 * \param[in] source_mr_index memory region index as passed to gpump_f_register_region from which data will be taken
 * \param[in] offset Offset in bytes from in the source_mr from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_stream_send(MPI_Fint vtarget, cudaStream_t vstream, MPI_Fint vsource_mr_index, gpump_size_t voffset, gpump_size_t vsize) ;
/**
 * \brief Nonblocking send for 2-sided communication triggered by CPU to communication partner
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] source_mr_index memory region index as passed to gpump_f_register_region from which data will be taken
 * \param[in] offset Offset in bytes from in the source_mr from which data will be taken
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_isend(MPI_Fint vtarget, MPI_Fint vsource_mr_index, gpump_size_t voffset, gpump_size_t vsize) ;
/**
 * \brief Block the stream until the send completes
 *
 * \param[in] target rank in communicator of communication partner
 * \param[in] stream CUDA stream to block until the send completes
 *
 */
void gpump_f_stream_wait_send_complete(MPI_Fint vtarget, cudaStream_t vstream) ;
/**
 * \brief CPU acknowledges that send will complete
 *
 * \note Make this call if you will not be making a call to gpump_stream_wait_send_complete for this transfer
 *
 * \param[in] target rank in communicator of communication partner
 */
void gpump_f_cpu_ack_isend(MPI_Fint vtarget) ;
/**
 * \brief Nonblocking test of whether the send is complete
 *
 * \note '1' will be returned once only for each send operation
 *
 * \param[in] target rank in communicator of communication partner
 * \param[out] is_complete 0 for send still in progress, 1 for complete
 *
 */
void gpump_f_is_send_complete(MPI_Fint vtarget, MPI_Fint *is_complete) ;
/**
 * \brief Blocking wait for send completion
 *
 * \note equivalent to calling gpump_f_is_send_complete in a loop until 1 is returned
 *
 * \param[in] target rank in communicator of communication partner
 *
 */
void gpump_f_wait_send_complete(MPI_Fint vtarget) ;
/**
 * \brief Nonblocking receive for 2-sided communication triggered by CPU to communication partner
 *
 * \note The underlying hardware does not support a 'gpump_f_stream_receive'
 *
 * \param[in] source rank in communicator of communication partner
 * \param[out] target_mr_index memory region as passed to gpump_f_register_region to which data will be written
 * \param[in] offset Offset in bytes from in the target_mr to which data will be written
 * \param[in] size Size in bytes of data transfer
 */
void gpump_f_receive(MPI_Fint vsource, MPI_Fint vtarget_mr_index, gpump_size_t voffset, gpump_size_t vsize) ;
/**
 * \brief Block the stream until the receive completes
 *
 * \param[in] source rank in communicator of communication partner
 * \param[in] stream CUDA stream to block until the receive completes
 *
 */
void gpump_f_stream_wait_recv_complete(MPI_Fint vsource, cudaStream_t vstream) ;
/**
 * \brief CPU acknowledges that receive will complete
 *
 * \note Make this call if you will not be making a call to gpump_stream_wait_recv_complete for this transfer
 *
 * \param[in] source rank in communicator of communication partner
 */
void gpump_f_cpu_ack_recv(MPI_Fint vsource) ;
/**
 * \brief Nonblocking test of whether the receive is complete
 *
 * \note '1' will be returned once only for each receive operation
 *
 * \param[in] target rank in communicator of communication partner
 *
 * \retval 0 The send is still in progress
 * \retval 1 The send is complete
 */
void gpump_f_is_receive_complete(MPI_Fint vtarget, MPI_Fint *is_complete) ;
/**
 * \brief Blocking wait for receive completion
 *
 * \note equivalent to calling gpump_is_receive_complete in a loop until 1 is returned
 *
 * \param[in] source rank in communicator of communication partner
 *
 */
void gpump_f_wait_receive_complete(MPI_Fint vsource) ;

/**
 * \brief Initialize the gpump library
 *
 * \note The Fortran interface is modelled after the C interface, with differences
 *  1) C pointers for memory regions are changed to array indices. The gpump library maintains the array.
 *  2) GPU addresses are to be passed by writing C_PTR(LOC(x)) in Fortran
 *  The gpump_f_init_r interface supports multiple communicators simultaneously.
 *  It is thread safe in respect of the first 10 communicators under the usual MPI restriction
 *  that a communicator is not used from omre than one thread concurrently, but is not thread safe
 *  beyond that
 *  becuase the C++ unordered map which goes from MPI Fortran communicator to internal gpump pointer
 *  is not thread safe. If the library is to be used with a threaded application and a lerger number
 *  of communicators, it is recommended that the constant  k_BaseCount in gpump_f.cpp is increased
 *  to handle the number of communicators that the application will use and that the library is
 *  recompiled.
 *
 * \param[in] comm The MPI communicator to use
 */
void gpump_f_init_r(MPI_Fint vcomm) ;
/**
 * \brief Terminate use of the gpump library
 *
 * \param[in] comm MPI communicator
 */
void gpump_f_term_r(MPI_Fint vcomm) ;

/**
 * \brief Register a region of GPU memory for use with the gpump library
 *
 * \param[in] comm MPI communicator
 * \param [out] mr_index Index to the memory region array for this region. Indices start at 1 to match the Fortran convention.
 * \param [in] addr address as returned by cudaMalloc
 * \param [in] size size in bytes of memory region
 *
 * \retval address of resulting struct ibv_mr describing the memory region
 *
 */
void gpump_f_register_region_r(MPI_Fint vcomm,MPI_Fint *mr_index,  void * vaddr, gpump_size_t vsize) ;
/**
 * \brief Replace a registered region of GPU memory for use with the gpump library
 *
 * \param[in] comm MPI communicator
 * \param [in] mr_index Index to the memory region array for this region. Indices start at 1 to match the Fortran convention.
 * \param [in] addr address as returned by cudaMalloc
 * \param [in] size size in bytes of memory region
 *
 * \retval address of resulting struct ibv_mr describing the memory region
 *
 */
void gpump_f_replace_region_r(MPI_Fint vcomm,MPI_Fint vmr_index,  void * vaddr, gpump_size_t vsize) ;
/**
 * \brief Deregister a region of GPU memory
 *
 * \param[in] comm MPI communicator
 * \param [in] mr_index Index to the memory region array for this region. Indices start at 1 to match the Fortran convention.
 *
 */
void gpump_f_deregister_region_r(MPI_Fint vcomm,MPI_Fint vmr_index) ;
/**
 * \brief Set the window count between 2 ranks
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] comm MPI communicator
 * \param [in] target rank in communicator of communication partner
 * \param [in] count number of window pairs for RDMA communication
 */
void gpump_f_set_window_count_r(MPI_Fint vcomm, MPI_Fint vtarget, MPI_Fint vcount) ;
/**
 * \brief Propose connection of 2 ranks for gpump data exchange
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] comm MPI communicator
 * \param [in] target rank in communicator of communication partner
 *
 */
void gpump_f_connect_propose_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
* \brief Accept connection of 2 ranks for gpump data exchange
*
* \note This function needs to be called by both communication partners. It is a blocking call.
*
 * \param[in] comm MPI communicator
* \param [in] target rank in communicator of communication partner
*
*/
void gpump_f_connect_accept_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
///**
// * \brief Connect 2 ranks for gpump data exchange
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param [in] target rank in communicator of communication partner
// *
// */
//void gpump_f_connect_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
* \brief Disconnect 2 ranks that have been connected for gpump data exchange
*
 * \param[in] comm MPI communicator
* \param [in] target rank in communicator of communication partner
*
*/
void gpump_f_disconnect_r(MPI_Fint vcomm,int target) ;
/**
 * \brief Propose creation of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] comm MPI communicator
 * \param[in] target rank in communicator of communication partner
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_f_create_window_propose_r(MPI_Fint vcomm,MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
 * \brief Propose replacement of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] comm MPI communicator
 * \param[in] target rank in communicator of communication partner
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_f_replace_window_propose_r(MPI_Fint vcomm,MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
* \brief Accept a pair of windows for RDMA put and get operations
*
* \note This function needs to be called by both communication partners. It is a blocking call.
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*
*/
void gpump_f_window_accept_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
///**
// * \brief Create a pair of windows for RDMA put and get operations
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param[in] target rank in communicator of communication partner
// * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
// * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
// * \param[in] size size in bytes of local and remote windows
// *
// */
//void gpump_f_create_window_r(MPI_Fint vcomm,MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
 * \brief Destroy the RDMA put/get window
 *
 * \note This function is usually used before establishing a new pair of windows with another gpump_create_window call.
 *
 * \param[in] comm MPI communicator
 * \param[in] target rank in communicator of communication partner
 */
void gpump_f_destroy_window_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
 * \brief Cork the put/get/send calls to the GPU. Put/get/send requests are deferred _r(MPI_Fint vcomm,batched) and sent to the GPU
 * on the uncork call
 * \param[in] comm MPI communicator
 */
/**
 * \brief Propose creation of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] comm MPI communicator
 * \param[in] target rank in communicator of communication partner
 * \param[out] index index in the per-window array
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_f_create_window_propose_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint *index, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
 * \brief Propose replacement of a pair of windows for RDMA put and get operations
 *
 * \note This function needs to be called by both communication partners.
 *
 * \param[in] comm MPI communicator
 * \param[in] target rank in communicator of communication partner
 * \param[in] index index in the per-window array
 * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
 * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
 * \param[in] size size in bytes of local and remote windows
 *
 */
void gpump_f_replace_window_propose_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
* \brief Accept creation of a pair of windows for RDMA put and get operations
*
* \note This function needs to be called by both communication partners. It is a blocking call.
*
* \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] index index in the per-window array
*
*/
void gpump_f_window_accept_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex) ;
///**
// * \brief Create a pair of windows for RDMA put and get operations
// *
// * \note This function needs to be called by both communication partners. It
// *       is a blocking call.
// *
// * \param[in] target rank in communicator of communication partner
// * \param[in] local_address address as returned by cudaMalloc of window used by put/get calls from this process
// * \param[in[ remote_address address as returned by cudaMalloc of window used by put/get calls from remote process
// * \param[in] size size in bytes of local and remote windows
// *
// */
//void gpump_f_create_window_r(MPI_Fint vcomm,MPI_Fint vtarget, void * vlocal_address, void * vremote_address, gpump_size_t vsize) ;
/**
 * \brief Destroy the RDMA put/get window
 *
 * \note This function is usually used before establishing a new pair of windows with another gpump_create_window call.
 *
 * \param[in] comm MPI communicator
 * \param[in] target rank in communicator of communication partner
 * \param[in] index index in the per-window array
 */
void gpump_f_destroy_window_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex) ;
/**
 * \brief Cork the put/get/send calls to the GPU. Put/get/send requests are deferred _r(MPI_Fint vcomm,batched) and sent to the GPU
 * on the uncork call
 * \param[in] comm MPI communicator
 */
void gpump_f_cork_r(MPI_Fint vcomm) ;
/**
 * \brief Uncork put/get/send calls to the GPU
 *
 * \param[in] stream CUDA stream which will initiate the requests to the HCA
 */
void gpump_f_uncork_r(MPI_Fint vcomm,cudaStream_t vstream) ;
/**
* \brief Nonblocking RDMA put on stream to communication partner
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream which will initiate the put request to the HCA
* \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
* \param[out] remote_offset Offset in bytes from the communication partner's ' remote address' in the window which data will be written to
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_stream_put_r(MPI_Fint vcomm,MPI_Fint vtarget, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
* \brief Nonblocking RDMA put triggered by CPU to communication partner
*
* \param[in] target rank in communicator of communication partner
* \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
* \param[out] remote_offset Offset in bytes from the communication partner's 'remote address' in the window which data will be written to
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_iput_r(MPI_Fint vcomm,MPI_Fint vtarget, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
* \brief Block the stream until the RDMA put completes
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream to block until the RDMA put completes
*
*/
void gpump_f_stream_wait_put_complete_r(MPI_Fint vcomm,MPI_Fint vtarget, cudaStream_t vstream) ;
/**
* \brief CPU acknowledges that RDMA put will complete
*
* \note Make this call if you will not be making a call to gpump_stream_wait_put_complete for this transfer
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*/
void gpump_f_cpu_ack_iput_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
* \brief Nonblocking test of whether the RDMA put is complete
*
* \note '1' will be returned once only for each RDMA put operation
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[out] is_complete 0 for still in progress, 1 for complete
*
*/
void gpump_f_is_put_complete_r(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint *is_complete) ;
/**
* \brief Blocking wait for RDMA completion
*
* \note equivalent to calling gpump_f_is_put_complete in a loop until 1 is returned
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*
*/
void gpump_f_wait_put_complete_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
* \brief Nonblocking RDMA get on stream from communication partner
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream which will initiate the get request to the HCA
* \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
* \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_stream_get_r(MPI_Fint vcomm,MPI_Fint vtarget, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
* \brief Nonblocking RDMA get triggered by CPU from communication partner
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
* \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_iget_r(MPI_Fint vcomm,MPI_Fint vtarget, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
* \brief Block the stream until the RDMA get completes
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream to block until the RDMA get completes
*
*/
void gpump_f_stream_wait_get_complete_r(MPI_Fint vcomm,MPI_Fint vtarget, cudaStream_t vstream) ;
/**
* \brief CPU acknowledges that RDMA get will complete
*
* \note Make this call if you will not be making a call to gpump_stream_wait_get_complete for this transfer
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*/
void gpump_f_cpu_ack_iget_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
* \brief Nonblocking test of whether the RDMA get is complete
*
* \note '1' will be returned once only for each RDMA get operation
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[out] is_complete 0 for still in progress, 1 for complete
*
*/
void gpump_f_is_get_complete_r(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint *is_complete) ;
/**
* \brief Blocking wait for RDMA completion
*
* \note equivalent to calling gpump_f_is_get_complete in a loop until 1 is returned
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*
*/
void gpump_f_wait_get_complete_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
* \brief Nonblocking RDMA put on stream to communication partner
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream which will initiate the put request to the HCA
* \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
* \param[out] remote_offset Offset in bytes from the communication partner's ' remote address' in the window which data will be written to
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_stream_put_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
* \brief Nonblocking RDMA put triggered by CPU to communication partner
*
* \param[in] target rank in communicator of communication partner
* \param[in] offset Offset in bytes from 'local address' in the window from which data will be taken
* \param[out] remote_offset Offset in bytes from the communication partner's 'remote address' in the window which data will be written to
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_iput_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
* \brief Block the stream until the RDMA put completes
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream to block until the RDMA put completes
*
*/
void gpump_f_stream_wait_put_complete_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, cudaStream_t vstream) ;
/**
* \brief CPU acknowledges that RDMA put will complete
*
* \note Make this call if you will not be making a call to gpump_stream_wait_put_complete for this transfer
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*/
void gpump_f_cpu_ack_iput_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex) ;
/**
* \brief Nonblocking test of whether the RDMA put is complete
*
* \note '1' will be returned once only for each RDMA put operation
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[out] is_complete 0 for still in progress, 1 for complete
*
*/
void gpump_f_is_put_complete_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, MPI_Fint *is_complete) ;
/**
* \brief Blocking wait for RDMA completion
*
* \note equivalent to calling gpump_f_is_put_complete in a loop until 1 is returned
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*
*/
void gpump_f_wait_put_complete_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex) ;
/**
* \brief Nonblocking RDMA get on stream from communication partner
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream which will initiate the get request to the HCA
* \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
* \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_stream_get_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, cudaStream_t vstream, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
* \brief Nonblocking RDMA get triggered by CPU from communication partner
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[out] offset Offset in bytes from 'local address' in the window to which data will be written
* \param[in] remote_offset Offset in bytes from the communication partner's 'remote address' in the window from which data will be taken
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_iget_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, gpump_size_t voffset, gpump_size_t vremote_offset, gpump_size_t vsize ) ;
/**
* \brief Block the stream until the RDMA get completes
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream to block until the RDMA get completes
*
*/
void gpump_f_stream_wait_get_complete_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, cudaStream_t vstream) ;
/**
* \brief CPU acknowledges that RDMA get will complete
*
* \note Make this call if you will not be making a call to gpump_stream_wait_get_complete for this transfer
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*/
void gpump_f_cpu_ack_iget_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex) ;
/**
* \brief Nonblocking test of whether the RDMA get is complete
*
* \note '1' will be returned once only for each RDMA get operation
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[out] is_complete 0 for still in progress, 1 for complete
*
*/
void gpump_f_is_get_complete_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex, MPI_Fint *is_complete) ;
/**
* \brief Blocking wait for RDMA completion
*
* \note equivalent to calling gpump_f_is_get_complete in a loop until 1 is returned
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*
*/
void gpump_f_wait_get_complete_rx(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vindex) ;
/**
* \brief Nonblocking send for 2-sided communication on stream to communication partner
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream which will initiate the send request to the HCA
* \param[in] source_mr_index memory region index as passed to gpump_f_register_region from which data will be taken
* \param[in] offset Offset in bytes from in the source_mr from which data will be taken
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_stream_send_r(MPI_Fint vcomm,MPI_Fint vtarget, cudaStream_t vstream, MPI_Fint vsource_mr_index, gpump_size_t voffset, gpump_size_t vsize) ;
/**
* \brief Nonblocking send for 2-sided communication triggered by CPU to communication partner
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] source_mr_index memory region index as passed to gpump_f_register_region from which data will be taken
* \param[in] offset Offset in bytes from in the source_mr from which data will be taken
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_isend_r(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint vsource_mr_index, gpump_size_t voffset, gpump_size_t vsize) ;
/**
* \brief Block the stream until the send completes
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[in] stream CUDA stream to block until the send completes
*
*/
void gpump_f_stream_wait_send_complete_r(MPI_Fint vcomm,MPI_Fint vtarget, cudaStream_t vstream) ;
/**
* \brief CPU acknowledges that send will complete
*
* \note Make this call if you will not be making a call to gpump_stream_wait_send_complete for this transfer
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*/
void gpump_f_cpu_ack_isend_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
* \brief Nonblocking test of whether the send is complete
*
* \note '1' will be returned once only for each send operation
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
* \param[out] is_complete 0 for send still in progress, 1 for complete
*
*/
void gpump_f_is_send_complete_r(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint *is_complete) ;
/**
* \brief Blocking wait for send completion
*
* \note equivalent to calling gpump_f_is_send_complete in a loop until 1 is returned
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*
*/
void gpump_f_wait_send_complete_r(MPI_Fint vcomm,MPI_Fint vtarget) ;
/**
* \brief Nonblocking receive for 2-sided communication triggered by CPU to communication partner
*
* \note The underlying hardware does not support a 'gpump_f_stream_receive'
*
 * \param[in] comm MPI communicator
* \param[in] source rank in communicator of communication partner
* \param[out] target_mr_index memory region as passed to gpump_f_register_region to which data will be written
* \param[in] offset Offset in bytes from in the target_mr to which data will be written
* \param[in] size Size in bytes of data transfer
*/
void gpump_f_receive_r(MPI_Fint vcomm,MPI_Fint vsource, MPI_Fint vtarget_mr_index, gpump_size_t voffset, gpump_size_t vsize) ;
/**
* \brief Block the stream until the receive completes
*
 * \param[in] comm MPI communicator
* \param[in] source rank in communicator of communication partner
* \param[in] stream CUDA stream to block until the receive completes
*
*/
void gpump_f_stream_wait_recv_complete_r(MPI_Fint vcomm,MPI_Fint vsource, cudaStream_t vstream) ;
/**
* \brief CPU acknowledges that receive will complete
*
* \note Make this call if you will not be making a call to gpump_stream_wait_recv_complete for this transfer
*
 * \param[in] comm MPI communicator
* \param[in] source rank in communicator of communication partner
*/
void gpump_f_cpu_ack_recv_r(MPI_Fint vcomm,MPI_Fint vsource) ;
/**
* \brief Nonblocking test of whether the receive is complete
*
* \note '1' will be returned once only for each receive operation
*
 * \param[in] comm MPI communicator
* \param[in] target rank in communicator of communication partner
*
* \retval 0 The send is still in progress
* \retval 1 The send is complete
*/
void gpump_f_is_receive_complete_r(MPI_Fint vcomm,MPI_Fint vtarget, MPI_Fint *is_complete) ;
/**
* \brief Blocking wait for receive completion
*
* \note equivalent to calling gpump_is_receive_complete in a loop until 1 is returned
*
 * \param[in] comm MPI communicator
* \param[in] source rank in communicator of communication partner
*
*/
void gpump_f_wait_receive_complete_r(MPI_Fint vcomm,MPI_Fint vsource) ;

#ifdef __cplusplus
}
#endif

#endif
