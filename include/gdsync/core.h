/* Copyright (c) 2016,2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __GDSYNC_CORE_H__
#define __GDSYNC_CORE_H__
#include <stdint.h>
#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>

enum gds_create_qp_flags {
    GDS_CREATE_QP_DEFAULT      = 0,
    GDS_CREATE_QP_WQ_ON_GPU    = 1<<0,
    GDS_CREATE_QP_TX_CQ_ON_GPU = 1<<1,
    GDS_CREATE_QP_RX_CQ_ON_GPU = 1<<2,
    GDS_CREATE_QP_WQ_DBREC_ON_GPU = 1<<5,
};

typedef struct ibv_exp_send_wr gds_send_wr;
typedef struct ibv_exp_qp_init_attr gds_qp_init_attr_t;

struct gds_cq {
        struct ibv_cq *cq;
        uint32_t curr_offset;
};

struct gds_qp {
        struct ibv_qp *qp;
        struct gds_cq send_cq;
        struct gds_cq recv_cq;
        struct ibv_exp_res_domain * res_domain;
        struct ibv_context *dev_context;
};

/* \brief: Create a peer-enabled QP attached to the specified GPU id.
 *
 * Peer QPs require dedicated send and recv CQs, e.g. cannot (easily)
 * use SRQ.
 */

struct gds_qp *gds_create_qp(struct ibv_pd *pd, struct ibv_context *context,
                             gds_qp_init_attr_t *qp_init_attr,
                             int gpu_id, int flags);

/* \brief: Destroy a peer-enabled QP
 *
 * The associated CQs are destroyed as well.
 */
int gds_destroy_qp(struct gds_qp *qp);


typedef enum gds_memory_type {
        GDS_MEMORY_GPU  = 1, /*< use this flag for both cudaMalloc/cuMemAlloc and cudaMallocHost/cuMemHostAlloc */
        GDS_MEMORY_HOST = 2,
        GDS_MEMORY_IO   = 4,
        GDS_MEMORY_MASK = 0x7
} gds_memory_type_t;

// Note: those flags below must not overlap with gds_memory_type_t
typedef enum gds_wait_flags {
        GDS_WAIT_POST_FLUSH_REMOTE = 1<<3, /*< add a trailing flush of the ingress GPUDirect RDMA data path on the GPU owning the stream */
        GDS_WAIT_POST_FLUSH = GDS_WAIT_POST_FLUSH_REMOTE /*< alias for backward compatibility */
} gds_wait_flags_t;

typedef enum gds_write_flags {
        GDS_WRITE_PRE_BARRIER_SYS = 1<<4, /*< add a heading memory barrier to the write value operation */
        GDS_WRITE_PRE_BARRIER = GDS_WRITE_PRE_BARRIER_SYS /*< alias for backward compatibility */
} gds_write_flags_t;

typedef enum gds_write_memory_flags {
        GDS_WRITE_MEMORY_POST_BARRIER_SYS = 1<<4, /*< add a trailing memory barrier to the memory write operation */
        GDS_WRITE_MEMORY_PRE_BARRIER_SYS  = 1<<5 /*< add a heading memory barrier to the memory write operation, for convenience only as not a native capability */
} gds_write_memory_flags_t;

typedef enum gds_membar_flags {
        GDS_MEMBAR_FLUSH_REMOTE = 1<<4,
        GDS_MEMBAR_DEFAULT      = 1<<5,
        GDS_MEMBAR_SYS          = 1<<6,
        GDS_MEMBAR_MLX5         = 1<<7 /*< modify the scope of the barrier, for internal use only */
} gds_membar_flags_t;

enum {
        GDS_SEND_INFO_MAX_OPS = 32,
        GDS_WAIT_INFO_MAX_OPS = 32
};

/**
 * Represents a posted send operation on a particular QP
 */

typedef struct gds_send_request {
        struct ibv_exp_peer_commit commit;
        struct peer_op_wr wr[GDS_SEND_INFO_MAX_OPS];
} gds_send_request_t;

//int gds_prepare_send(struct gds_qp *qp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr, gds_send_request_t *request);
//int gds_stream_post_send(CUstream stream, gds_send_request_t *request);
//int gds_stream_post_send_all(CUstream stream, int count, gds_send_request_t *request);

/**
 * Represents a wait operation on a particular CQ
 */

typedef struct gds_wait_request {
        struct ibv_exp_peer_peek peek;
        struct peer_op_wr wr[GDS_WAIT_INFO_MAX_OPS];
} gds_wait_request_t;

/**
 * Initializes a wait request out of the next heading CQE, which is kept in
 * cq->curr_offset.
 *
 * flags: must be 0
 */
//int gds_prepare_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags);

/**
 * Issues the descriptors contained in request on the CUDA stream
 *
 */
//int gds_stream_post_wait_cq(CUstream stream, gds_wait_request_t *request);

/**
 * Issues the descriptors contained in the array of requests on the CUDA stream.
 * This has potentially less overhead than submitting each request individually.
 *
 */
//int gds_stream_post_wait_cq_all(CUstream stream, int count, gds_wait_request_t *request);

/**
 * \brief CPU-synchronously enable polling on request
 *
 * Unblock calls to ibv_poll_cq. CPU will do what is necessary to make the corresponding
 * CQE poll-able.
 *
 */
//int gds_post_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags);

/**
 * Represents the condition operation for wait operations on memory words
 */

typedef enum gds_wait_cond_flag {
        GDS_WAIT_COND_GEQ = 0, // must match verbs_exp enum
        GDS_WAIT_COND_EQ,
        GDS_WAIT_COND_AND,
        GDS_WAIT_COND_NOR
} gds_wait_cond_flag_t;

/**
 * Represents a wait operation on a 32-bits memory word
 */

typedef struct gds_wait_value32 {
        uint32_t  *ptr;
        uint32_t   value;
        gds_wait_cond_flag_t cond_flags;
        int        flags; // takes gds_memory_type_t | gds_wait_flags_t
} gds_wait_value32_t;

/**
 * flags: gds_memory_type_t | gds_wait_flags_t
 */
//int gds_prepare_wait_value32(gds_wait_value32_t *desc, uint32_t *ptr, uint32_t value, gds_wait_cond_flag_t cond_flags, int flags);



/**
 * Represents a write operation on a 32-bits memory word
 */

typedef struct gds_write_value32 {
        uint32_t  *ptr;
        uint32_t   value;
        int        flags; // takes gds_memory_type_t | gds_write_flags_t
} gds_write_value32_t;

/**
 * flags:  gds_memory_type_t | gds_write_flags_t
 */
//int gds_prepare_write_value32(gds_write_value32_t *desc, uint32_t *ptr, uint32_t value, int flags);



/**
 * Represents a staged copy operation
 * the src buffer can be reused after the API call
 */

typedef struct gds_write_memory {
        uint8_t       *dest;
        const uint8_t *src;
        size_t         count;
        int            flags; // takes gds_memory_type_t | gds_write_memory_flags_t
} gds_write_memory_t;

/**
 * flags:  gds_memory_type_t | gds_write_memory_flags_t
 */
//int gds_prepare_write_memory(gds_write_memory_t *desc, uint8_t *dest, const uint8_t *src, size_t count, int flags);



typedef enum gds_tag {
        GDS_TAG_SEND,
        GDS_TAG_WAIT,
        GDS_TAG_WAIT_VALUE32,
        GDS_TAG_WRITE_VALUE32,
        GDS_TAG_WRITE_MEMORY
} gds_tag_t;

typedef struct gds_descriptor {
        gds_tag_t tag; /**< selector for union below */
        union {
                gds_send_request_t  *send;
                gds_wait_request_t  *wait;
                gds_wait_value32_t   wait32;
                gds_write_value32_t  write32;
                gds_write_memory_t   writemem;
        };
} gds_descriptor_t;

#endif
