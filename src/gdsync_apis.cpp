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

#include <unistd.h>
#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "config.h"
#include "gdsync/core.h"
#include "gdsync_utils.h"
#include "gdsync_objs.h"
#include "gpump_cuda_wrapper.h"

static void gds_init_ops(struct peer_op_wr *op, int count)
{
        int i = count;
        while (--i)
                op[i-1].next = &op[i];
        op[count-1].next = NULL;
}

//-----------------------------------------------------------------------------

void gds_init_send_info(gds_send_request_t *info)
{
        gds_dbg("send_request=%p\n", info);
        memset(info, 0, sizeof(*info));

        info->commit.storage = info->wr;
        info->commit.entries = sizeof(info->wr)/sizeof(info->wr[0]);
        gds_init_ops(info->commit.storage, info->commit.entries);
}

//-----------------------------------------------------------------------------

static void gds_init_wait_request(gds_wait_request_t *request, uint32_t offset)
{
        gds_dbg("wait_request=%p offset=%08x\n", request, offset);
        memset(request, 0, sizeof(*request));
        request->peek.storage = request->wr;
        request->peek.entries = sizeof(request->wr)/sizeof(request->wr[0]);
        request->peek.whence = IBV_EXP_PEER_PEEK_ABSOLUTE;
        request->peek.offset = offset;
        gds_init_ops(request->peek.storage, request->peek.entries);
}

//-----------------------------------------------------------------------------

int gds_prepare_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags)
{
        int retcode = 0;
        if (flags != 0) {
                gds_err("invalid flags != 0\n");
                return EINVAL;
        }

        gds_init_wait_request(request, cq->curr_offset++);

        retcode = ibv_exp_peer_peek_cq(cq->cq, &request->peek);
        if (retcode == -ENOSPC) {
                // TODO: handle too few entries
                gds_err("not enough ops in peer_peek_cq\n");
                goto out;
        } else if (retcode) {
                gds_err("error %d in peer_peek_cq\n", retcode);
                goto out;
        }
        //gds_dump_wait_request(request, 1);
        out:
               return retcode;
}

//-----------------------------------------------------------------------------

int gds_stream_post_wait_cq(CUstream stream, gds_wait_request_t *request)
{
        return gds_stream_post_wait_cq_multi(stream, 1, request, NULL, 0);
}

//-----------------------------------------------------------------------------

int gds_cork_post_wait_cq(CUstream stream, gds_op_list_t &params, gds_wait_request_t *request)
{
        return gds_cork_post_wait_cq_multi(stream, params, 1, request, NULL, 0);
}

//-----------------------------------------------------------------------------

static bool no_network_descs_after_entry(size_t n_descs, gds_descriptor_t *descs, size_t idx)
{
        bool ret = true;
        size_t i;
        for(i = idx+1; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_SEND:
                case GDS_TAG_WAIT:
                        ret = false;
                        goto out;
                case GDS_TAG_WAIT_VALUE32:
                case GDS_TAG_WRITE_VALUE32:
                        break;
                default:
                        gds_err("invalid tag\n");
                        ret = EINVAL;
                        goto out;
                }
        }
out:
        return ret;
}

static int get_wait_info(size_t n_descs, gds_descriptor_t *descs, size_t &n_waits, size_t &last_wait)
{
        int ret = 0;
        size_t i;
        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_WAIT:
                        ++n_waits;
                        last_wait = i;
                        break;
                case GDS_TAG_SEND:
                case GDS_TAG_WAIT_VALUE32:
                case GDS_TAG_WRITE_VALUE32:
                case GDS_TAG_WRITE_MEMORY:
                        break;
                default:
                        gds_err("invalid tag\n");
                        ret = EINVAL;
                }
        }
        return ret;
}

static int calc_n_mem_ops(size_t n_descs, gds_descriptor_t *descs, size_t &n_mem_ops)
{
        int ret = 0;
        n_mem_ops = 0;
        size_t i;
        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_SEND:
                        n_mem_ops += desc->send->commit.entries + 2; // extra space, ugly
                        break;
                case GDS_TAG_WAIT:
                        n_mem_ops += desc->wait->peek.entries + 2; // ditto
                        break;
                case GDS_TAG_WAIT_VALUE32:
                case GDS_TAG_WRITE_VALUE32:
                case GDS_TAG_WRITE_MEMORY:
                        n_mem_ops += 2; // ditto
                        break;
                default:
                        gds_err("invalid tag\n");
                        ret = EINVAL;
                }
        }
        return ret;
}
int gds_stream_post_descriptors(CUstream stream, size_t n_descs, gds_descriptor_t *descs, int flags)
{
        size_t i;
        int idx = 0;
        int ret = 0;
        int retcode = 0;
        size_t n_mem_ops = 0;
        size_t n_waits = 0;
        size_t last_wait = 0;
        bool move_flush = false;
        gds_peer *peer = NULL;
        gds_op_list_t params;


        ret = calc_n_mem_ops(n_descs, descs, n_mem_ops);
        if (ret) {
                gds_err("error %d in calc_n_mem_ops\n", ret);
                goto out;
        }

        ret = get_wait_info(n_descs, descs, n_waits, last_wait);
        if (ret) {
                gds_err("error %d in get_wait_info\n", ret);
                goto out;
        }

        gds_dbg("n_descs=%zu n_waits=%zu n_mem_ops=%zu\n", n_descs, n_waits, n_mem_ops);

        // move flush to last wait in the whole batch
        if (n_waits && no_network_descs_after_entry(n_descs, descs, last_wait)) {
                gds_dbg("optimizing FLUSH to last wait i=%zu\n", last_wait);
                move_flush = true;
        }
        // alternatively, remove flush for wait is next op is a wait too

        peer = peer_from_stream(stream);
        if (!peer) {
                return EINVAL;
        }

        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_SEND: {
                        gds_send_request_t *sreq = desc->send;
                        retcode = gds_post_ops(peer, sreq->commit.entries, sreq->commit.storage, params);
                        if (retcode) {
                                gds_err("error %d in gds_post_ops\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                }
                case GDS_TAG_WAIT: {
                        gds_wait_request_t *wreq = desc->wait;
                        int flags = 0;
                        if (move_flush && i != last_wait) {
                                gds_dbg("discarding FLUSH!\n");
                                flags = GDS_POST_OPS_DISCARD_WAIT_FLUSH;
                        }
                        retcode = gds_post_ops(peer, wreq->peek.entries, wreq->peek.storage, params, flags);
                        if (retcode) {
                                gds_err("error %d in gds_post_ops\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                }
                case GDS_TAG_WAIT_VALUE32:
                        retcode = gds_fill_poll(peer, params, desc->wait32.ptr, desc->wait32.value, desc->wait32.cond_flags, desc->wait32.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_poll\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                case GDS_TAG_WRITE_VALUE32:
                        retcode = gds_fill_poke(peer, params, desc->write32.ptr, desc->write32.value, desc->write32.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_poke\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                case GDS_TAG_WRITE_MEMORY:
                        retcode = gds_fill_inlcpy(peer, params, desc->writemem.dest, desc->writemem.src, desc->writemem.count, desc->writemem.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_inlcpy\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                default:
                        gds_err("invalid tag for %zu entry\n", i);
                        ret = EINVAL;
                        goto out;
                        break;
                }
        }
        retcode = gds_stream_batch_ops(peer, stream, params, 0);
        if (retcode) {
                gds_err("error %d in gds_stream_batch_ops\n", retcode);
                ret = retcode;
                goto out;
        }

out:
        return ret;
}


int gds_cork_post_descriptors(CUstream stream, gds_op_list_t &params, size_t n_descs, gds_descriptor_t *descs, int flags)
{
        size_t i;
        int idx = 0;
        int ret = 0;
        int retcode = 0;
        size_t n_mem_ops = 0;
        size_t n_waits = 0;
        size_t last_wait = 0;
        bool move_flush = false;
        gds_peer *peer = NULL;


        ret = calc_n_mem_ops(n_descs, descs, n_mem_ops);
        if (ret) {
                gds_err("error %d in calc_n_mem_ops\n", ret);
                goto out;
        }

        ret = get_wait_info(n_descs, descs, n_waits, last_wait);
        if (ret) {
                gds_err("error %d in get_wait_info\n", ret);
                goto out;
        }

        gds_dbg("n_descs=%zu n_waits=%zu n_mem_ops=%zu\n", n_descs, n_waits, n_mem_ops);

        // move flush to last wait in the whole batch
        if (n_waits && no_network_descs_after_entry(n_descs, descs, last_wait)) {
                gds_dbg("optimizing FLUSH to last wait i=%zu\n", last_wait);
                move_flush = true;
        }
        // alternatively, remove flush for wait is next op is a wait too

        peer = peer_from_stream(stream);
        if (!peer) {
                return EINVAL;
        }

        for(i = 0; i < n_descs; ++i) {
                gds_descriptor_t *desc = descs + i;
                switch(desc->tag) {
                case GDS_TAG_SEND: {
                        gds_send_request_t *sreq = desc->send;
                        retcode = gds_post_ops(peer, sreq->commit.entries, sreq->commit.storage, params);
                        if (retcode) {
                                gds_err("error %d in gds_post_ops\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                }
                case GDS_TAG_WAIT: {
                        gds_wait_request_t *wreq = desc->wait;
                        int flags = 0;
                        if (move_flush && i != last_wait) {
                                gds_dbg("discarding FLUSH!\n");
                                flags = GDS_POST_OPS_DISCARD_WAIT_FLUSH;
                        }
                        retcode = gds_post_ops(peer, wreq->peek.entries, wreq->peek.storage, params, flags);
                        if (retcode) {
                                gds_err("error %d in gds_post_ops\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                }
                case GDS_TAG_WAIT_VALUE32:
                        retcode = gds_fill_poll(peer, params, desc->wait32.ptr, desc->wait32.value, desc->wait32.cond_flags, desc->wait32.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_poll\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                case GDS_TAG_WRITE_VALUE32:
                        retcode = gds_fill_poke(peer, params, desc->write32.ptr, desc->write32.value, desc->write32.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_poke\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                case GDS_TAG_WRITE_MEMORY:
                        retcode = gds_fill_inlcpy(peer, params, desc->writemem.dest, desc->writemem.src, desc->writemem.count, desc->writemem.flags);
                        if (retcode) {
                                gds_err("error %d in gds_fill_inlcpy\n", retcode);
                                ret = retcode;
                                goto out;
                        }
                        break;
                default:
                        gds_err("invalid tag for %zu entry\n", i);
                        ret = EINVAL;
                        goto out;
                        break;
                }
        }
//        retcode = gds_stream_batch_ops(peer, stream, params, 0);
//        if (retcode) {
//                gds_err("error %d in gds_stream_batch_ops\n", retcode);
//                ret = retcode;
//                goto out;
//        }

out:
        return ret;
}

int gds_uncork_post_descriptors(CUstream stream, gds_op_list_t &params)
  {
    int ret=0 ;
    gds_peer *peer = peer_from_stream(stream);
    if ( ! peer)
      {
        ret = EINVAL ;
        goto out ;
      }
    ret = gds_stream_batch_ops(peer, stream, params, 0);
    if ( ret )
      {
        gds_err("error %d in gds_stream_batch_ops\n", ret) ;
      }
out:
    params.clear() ;
    return ret ;
  }

//-----------------------------------------------------------------------------

int gds_prepare_send(struct gds_qp *qp, gds_send_wr *p_ewr,
                     gds_send_wr **bad_ewr,
                     gds_send_request_t *request)
{
        int ret = 0;
        gds_init_send_info(request);
        assert(qp);
        assert(qp->qp);
        ret = ibv_exp_post_send(qp->qp, p_ewr, bad_ewr);
        if (ret) {

                if (ret == ENOMEM) {
                        // out of space error can happen too often to report
                        gds_dbg("ENOMEM error %d in ibv_exp_post_send\n", ret);
                } else {
                        gds_err("error %d in ibv_exp_post_send\n", ret);
                }
                goto out;
        }

        ret = ibv_exp_peer_commit_qp(qp->qp, &request->commit);
        if (ret) {
                gds_err("error %d in ibv_exp_peer_commit_qp\n", ret);
                //gds_wait_kernel();
                goto out;
        }
out:
        return ret;
}

//-----------------------------------------------------------------------------

static int gds_rollback_qp(struct gds_qp *qp, gds_send_request_t * send_info, enum ibv_exp_rollback_flags flag)
{
        struct ibv_exp_rollback_ctx rollback;
        int ret=0;

        assert(qp);
        assert(qp->qp);
        assert(send_info);
        if(
                        flag != IBV_EXP_ROLLBACK_ABORT_UNCOMMITED &&
                        flag != IBV_EXP_ROLLBACK_ABORT_LATE
          )
        {
                gds_err("erroneous ibv_exp_rollback_flags flag input value\n");
                ret=EINVAL;
                goto out;
        }

        /* from ibv_exp_peer_commit call */
        rollback.rollback_id = send_info->commit.rollback_id;
        /* from ibv_exp_rollback_flag */
        rollback.flags = flag;
        /* Reserved for future expensions, must be 0 */
        rollback.comp_mask = 0;
        gds_warn("Need to rollback WQE %lx\n", rollback.rollback_id);
        ret = ibv_exp_rollback_qp(qp->qp, &rollback);
        if(ret)
                gds_err("error %d in ibv_exp_rollback_qp\n", ret);

out:
        return ret;
}

//-----------------------------------------------------------------------------

int gds_post_send(struct gds_qp *qp, gds_send_wr *p_ewr, gds_send_wr **bad_ewr)
{
        int ret = 0, ret_roll=0;
        gds_send_request_t send_info;
        ret = gds_prepare_send(qp, p_ewr, bad_ewr, &send_info);
        if (ret) {
                gds_err("error %d in gds_prepare_send\n", ret);
                goto out;
        }

        ret = gds_post_pokes_on_cpu(1, &send_info, NULL, 0);
        if (ret) {
                gds_err("error %d in gds_post_pokes_on_cpu\n", ret);
                ret_roll = gds_rollback_qp(qp, &send_info, IBV_EXP_ROLLBACK_ABORT_LATE);
                if (ret_roll) {
                        gds_err("error %d in gds_rollback_qp\n", ret_roll);
                }

                goto out;
        }

out:
        return ret;
}

//-----------------------------------------------------------------------------

static int gds_abort_wait_cq(struct gds_cq *cq, gds_wait_request_t *request)
{
        assert(cq);
        assert(request);
        struct ibv_exp_peer_abort_peek abort_ctx;
        abort_ctx.peek_id = request->peek.peek_id;
        abort_ctx.comp_mask = 0;
        return ibv_exp_peer_abort_peek_cq(cq->cq, &abort_ctx);
}

//-----------------------------------------------------------------------------

int gds_post_wait_cq(struct gds_cq *cq, gds_wait_request_t *request, int flags)
{
        int retcode = 0;

        if (flags) {
                retcode = EINVAL;
                goto out;
        }

        retcode = gds_abort_wait_cq(cq, request);
out:
        return retcode;
}


