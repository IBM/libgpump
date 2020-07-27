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

#include <assert.h>
#include <unistd.h>

#include "config.h"
#include "gdsync/core.h"
#include "gdsync_mem.h"
#include "gdsync_memmgr.h"
#include "gdsync_utils.h"
#include "gpump_cuda_wrapper.h"

typedef struct gds_mem_desc {
    CUdeviceptr d_ptr;
    void       *h_ptr;
    void       *bar_ptr;
    int         flags;
    size_t      alloc_size;
//    gdr_mh_t    mh;
} gds_mem_desc_t;

//-----------------------------------------------------------------------------

static int gds_alloc_pinned_memory(gds_mem_desc_t *desc, size_t size, int flags)
{
        int ret;
#ifdef USE_STATIC_MEM
        unsigned long s_addr = (unsigned long)(s_buf + s_buf_i);
        unsigned long s_off  = s_addr & (PAGE_SIZE-1);
        if (s_off)
                s_addr += PAGE_SIZE - s_off;
        *memptr = (void *)s_addr;
        gpu_info("s_buf_i=%d off=%lu memptr=%p\n", s_buf_i, s_off, *memptr);

        s_buf_i += s_off + size;

        if (s_buf_i >= s_size) {
                gds_err("can't alloc static memory\n");
                return ENOMEM;
        }
#else
        assert(desc);
        desc->h_ptr = NULL;
        ret = posix_memalign(&desc->h_ptr, GDS_HOST_PAGE_SIZE, size);
        if (ret) {
                goto out;
        }
        desc->d_ptr = 0;
        ret = gds_register_mem(desc->h_ptr, size, GDS_MEMORY_HOST, &desc->d_ptr);
        if (ret) {
                goto out;
        }
        desc->bar_ptr = NULL;
        desc->flags = flags;
        desc->alloc_size = size;
//        desc->mh = 0;
        gds_dbg("d_ptr=%lx h_ptr=%p flags=0x%08x alloc_size=%zd\n",
                (unsigned long)desc->d_ptr, desc->h_ptr, desc->flags, desc->alloc_size);
out:
        if (ret) {
                if (desc->h_ptr) {
                        if (desc->d_ptr)
                                gds_unregister_mem(desc->h_ptr, desc->alloc_size);
                        free(desc->h_ptr);
                }
        }
#endif
        return ret;
}

//-----------------------------------------------------------------------------

static int gds_free_pinned_memory(gds_mem_desc_t *desc)
{
        int ret;
        assert(desc);
        if (!desc->d_ptr || !desc->h_ptr) {
                gds_err("invalid desc\n");
                return EINVAL;
        }
#ifdef USE_STATIC_MEM
        // BUG: TBD
#else
        gds_dbg("d_ptr=%lx h_ptr=%p flags=0x%08x alloc_size=%zd\n",
                (unsigned long)desc->d_ptr, desc->h_ptr, desc->flags, desc->alloc_size);
        ret = gds_unregister_mem(desc->h_ptr, desc->alloc_size);
        free(desc->h_ptr);
        desc->h_ptr = NULL;
        desc->d_ptr = 0;
        desc->alloc_size = 0;
#endif
        return ret;
}

//-----------------------------------------------------------------------------

int gds_alloc_mapped_memory(gds_mem_desc_t *desc, size_t size, int flags)
{
        int ret = 0;
        if (!size) {
                gds_warn("silently ignoring zero size alloc!\n");
                return 0;
        }
        if (!desc) {
                gds_err("NULL desc!\n");
                return EINVAL;
        }
        switch(flags & GDS_MEMORY_MASK) {
        case GDS_MEMORY_GPU:
//                ret = gds_alloc_gdr_memory(desc, size, flags);
                ret = ENOSYS ;
                break;
        case GDS_MEMORY_HOST:
                ret = gds_alloc_pinned_memory(desc, size, flags);
                break;
        default:
                gds_err("invalid flags\n");
                ret = EINVAL;
                break;
        }
        return ret;
}

//-----------------------------------------------------------------------------

int gds_free_mapped_memory(gds_mem_desc_t *desc)
{
        int ret = 0;
        if (!desc) {
                gds_err("NULL desc!\n");
                return EINVAL;
        }
        switch(desc->flags & GDS_MEMORY_MASK) {
        case GDS_MEMORY_GPU:
//                ret = gds_free_gdr_memory(desc);
                ret = ENOSYS ;
                break;
        case GDS_MEMORY_HOST:
                ret = gds_free_pinned_memory(desc);
                break;
        default:
                ret = EINVAL;
                break;
        }
        return ret;
}

#define ROUND_TO(V,PS) ((((V) + (PS) - 1)/(PS)) * (PS))
//#define ROUND_TO_GDR_GPU_PAGE(V) ROUND_TO(V, GDR_GPU_PAGE_SIZE)

// allocate GPU memory with a GDR mapping (CPU can dereference it)
int gds_peer_malloc_ex(int peer_id, uint64_t peer_data, void **host_addr, CUdeviceptr *peer_addr, size_t req_size, void **phandle, bool has_cpu_mapping)
{
        int ret = 0;
        // assume GPUs are the only peers!!!
        int gpu_id = peer_id;
        CUcontext gpu_ctx;
        CUdevice gpu_device;
        size_t size = ROUND_TO(req_size, GDS_GPU_PAGE_SIZE);

        gds_dbg("GPU%u: malloc req_size=%zu size=%zu\n", gpu_id, req_size, size);

        if (!phandle || !host_addr || !peer_addr) {
                gds_err("invalid params\n");
                return EINVAL;
        }

        // NOTE: gpu_id's primary context is assumed to be the right one
        // breaks horribly with multiple contexts
        int num_gpus;
        do {
                CUresult err = cuDeviceGetCount(&num_gpus);
                if (CUDA_SUCCESS == err) {
                        break;
                } else if (CUDA_ERROR_NOT_INITIALIZED == err) {
                        gds_err("CUDA error %d in cuDeviceGetCount, calling cuInit\n", err);
                        CUCHECK(cuInit(0));
                        // try again
                        continue;
                } else {
                        gds_err("CUDA error %d in cuDeviceGetCount, returning EIO\n", err);
                        return EIO;
                }
        } while(0);
        gds_dbg("num_gpus=%d\n", num_gpus);
        if (gpu_id >= num_gpus) {
                gds_err("invalid num_GPUs=%d while requesting GPU id %d\n", num_gpus, gpu_id);
                return EINVAL;
        }

        CUCHECK(cuDeviceGet(&gpu_device, gpu_id));
        gds_dbg("gpu_id=%d gpu_device=%d\n", gpu_id, gpu_device);
        // TODO: check for existing context before switching to the interop one
        CUCHECK(cuDevicePrimaryCtxRetain(&gpu_ctx, gpu_device));
        CUCHECK(cuCtxPushCurrent(gpu_ctx));
        assert(gpu_ctx != NULL);

        gds_mem_desc_t *desc = (gds_mem_desc_t *)calloc(1, sizeof(gds_mem_desc_t));
        if (!desc) {
                gds_err("error while allocating mem desc\n");
                ret = ENOMEM;
                goto out;
        }

        ret = gds_alloc_mapped_memory(desc, size, GDS_MEMORY_GPU);
        if (ret) {
                gds_err("error %d while allocating mapped GPU buffers\n", ret);
                goto out;
        }

        *host_addr = desc->h_ptr;
        *peer_addr = desc->d_ptr;
        *phandle = desc;

out:
        if (ret)
                free(desc); // desc can be NULL

        CUCHECK(cuCtxPopCurrent(NULL));
        CUCHECK(cuDevicePrimaryCtxRelease(gpu_device));

        return ret;
}

//-----------------------------------------------------------------------------

int gds_peer_malloc(int peer_id, uint64_t peer_data, void **host_addr, CUdeviceptr *peer_addr, size_t req_size, void **phandle)
{
        return gds_peer_malloc_ex(peer_id, peer_data, host_addr, peer_addr, req_size, phandle, true);
}

//-----------------------------------------------------------------------------

int gds_peer_mfree(int peer_id, uint64_t peer_data, void *host_addr, void *handle)
{
        int ret = 0;
        // assume GPUs are the only peers!!!
        int gpu_id = peer_id;
        CUcontext gpu_ctx;
        CUdevice gpu_device;

        gds_dbg("GPU%u: mfree\n", gpu_id);

        if (!handle) {
                gds_err("invalid handle\n");
                return EINVAL;
        }

        if (!host_addr) {
                gds_err("invalid host_addr\n");
                return EINVAL;
        }

        // NOTE: gpu_id's primary context is assumed to be the right one
        // breaks horribly with multiple contexts

        CUCHECK(cuDeviceGet(&gpu_device, gpu_id));
        CUCHECK(cuDevicePrimaryCtxRetain(&gpu_ctx, gpu_device));
        CUCHECK(cuCtxPushCurrent(gpu_ctx));
        assert(gpu_ctx != NULL);

        gds_mem_desc_t *desc = (gds_mem_desc_t *)handle;
        ret = gds_free_mapped_memory(desc);
        if (ret) {
                gds_err("error %d while freeing mapped GPU buffers\n", ret);
        }
        free(desc);

        CUCHECK(cuCtxPopCurrent(NULL));
        CUCHECK(cuDevicePrimaryCtxRelease(gpu_device));

        return ret;
}
