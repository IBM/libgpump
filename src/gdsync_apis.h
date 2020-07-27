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

#ifndef __GDSYNC_APIS_H__
#define __GDSYNC_APIS_H__
#include "gdsync_objs.h"

void gds_init_send_info(gds_send_request_t *info);
int gds_prepare_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags);
int gds_stream_post_wait_cq(CUstream stream, gds_wait_request_t *request);
int gds_cork_post_wait_cq(CUstream stream, gds_op_list_t &params, gds_wait_request_t *request) ;
int gds_stream_post_descriptors(CUstream stream, size_t n_descs, gds_descriptor_t *descs, int flags);
int gds_cork_post_descriptors(CUstream stream, gds_op_list_t & params, size_t n_descs, gds_descriptor_t *descs, int flags);
int gds_uncork_post_descriptors(CUstream stream, gds_op_list_t &params) ;
int gds_register_peer_by_ordinal(unsigned gpu_id, gds_peer **p_peer, gds_peer_attr **p_peer_attr);
int gds_post_send(struct gds_qp *qp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr) ;
int gds_post_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags) ;

#endif
