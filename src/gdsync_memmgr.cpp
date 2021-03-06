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
#include <string.h>
#include <assert.h>
#include <inttypes.h>

#include <map>
#include <algorithm>
#include <string>
using namespace std;

#include <cuda.h>

#include "config.h"
#include "gdsync/core.h"
#include "gdsync_objs.h"
#include "gdsync_utils.h"
#include "gdsync_rangeset.h"
#include "gpump_cuda_wrapper.h"

//-----------------------------------------------------------------------------
// pin-down cache
//--------------
typedef Range range;
typedef RangeSet range_set;
static range_set rset;

typedef std::map<unsigned long, CUdeviceptr> pindown_cache_t;
static pindown_cache_t pinned_ranges;

static int gds_register_mem_internal(void *ptr, size_t size, gds_memory_type_t type, CUdeviceptr *dev_ptr);

// map whole pages contained in [ptr,ptr+size)
// return the CUdeviceptr corresponding to ptr
// BUG: after destroying a GPU context, all the GPU mappings will be invalidated
//      but this is not reflected here
// BUG: convert CUCHECK into error checks and return an error

                // loop over overlapping ranges,
                //     maybe even on consecutive ranges when not on page boundary
                //   unregister range
                //   merge with union range
                // register union range

//-----------------------------------------------------------------------------

int gds_map_mem(void *ptr, size_t size, gds_memory_type_t mem_type, CUdeviceptr *dev_ptr)
{
        assert(dev_ptr);

        gds_dbg("ptr=%p size=%zu mem_type=%08x\n", ptr, size, mem_type);

        range r((ptrdiff_t)ptr, (ptrdiff_t)ptr + size -1);

        range_set::find_result res = rset.find(r);
        switch(res.second) {
        case range_set::not_found:
                return gds_register_mem_internal(ptr, size, mem_type, dev_ptr);
                break;
        case range_set::partial_overlap:
                gds_err("partial overlap, buffer already registered?\n");
                return EINVAL;
        case range_set::fully_contained: {
                range r = *res.first;
                if (dev_ptr) {
                        pindown_cache_t::iterator found = pinned_ranges.find(r.first);
                        if (found != pinned_ranges.end()) {
                                CUdeviceptr page_dev_ptr = (*found).second;
                                ptrdiff_t off = (ptrdiff_t)ptr - (ptrdiff_t)r.first;
                                *dev_ptr = page_dev_ptr + off;
                        }
                        else {
                                gds_err("can't find dev_ptr for page_addr=%lx\n", r.first);
                                return EINVAL;
                        }
                }
                break;
        }
        default:
                gds_err("unexpected result");
                return EINVAL;
        }

        return 0;

}

//-----------------------------------------------------------------------------

int gds_register_mem(void *ptr, size_t size, gds_memory_type_t mem_type, CUdeviceptr *dev_ptr)
{
        return gds_map_mem(ptr, size, mem_type, dev_ptr);
}

int gds_register_mem_internal(void *ptr, size_t size, gds_memory_type_t type, CUdeviceptr *dev_ptr)
{
        gds_dbg("ptr=%p size=%zu memtype=%d\n", ptr, size, type);
        unsigned long addr = (unsigned long)ptr;
        CUdeviceptr page_dev_ptr = 0;
        // NOTE: registering buffer address on GPU MMU, on all GPUs
        unsigned int flags = CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_PORTABLE;
        bool cuda_registered = false;
        bool need_cuda_registration = true;
        unsigned long target_page_mask = 0;
        unsigned long target_page_off = 0;
        unsigned long target_page_size = 0;

        switch (type) {
        case GDS_MEMORY_GPU:
                gds_dbg("this is GPU memory, no CUDA registration required\n");
                need_cuda_registration = false;
                target_page_mask = GDS_GPU_PAGE_MASK;
                target_page_off = GDS_GPU_PAGE_OFF;
                target_page_size = GDS_GPU_PAGE_SIZE;
                break;
        case GDS_MEMORY_IO:
                flags |= CU_MEMHOSTREGISTER_IOMEMORY;
                // fall through
        case GDS_MEMORY_HOST:
                target_page_mask = GDS_HOST_PAGE_MASK;
                target_page_off = GDS_HOST_PAGE_OFF;
                target_page_size = GDS_HOST_PAGE_SIZE;
                break;
        default:
                gds_err("invalid mem type %d\n", type);
                return EINVAL;
        }

        unsigned long page_addr = addr & target_page_mask;
        unsigned long page_off = addr & target_page_off;
        size_t len = ROUND_UP(size + page_off, target_page_size);

        if (need_cuda_registration) {
                gds_dbg("calling cuMemHostRegister(%p, %zu, 0x%x)\n", (void*)page_addr, len, flags);
                CUresult res = cuMemHostRegister((void*)page_addr, len, flags);
                if (res == CUDA_SUCCESS) {
                        // we are good here
                }
                else if ((res == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) ||
                         (res == CUDA_ERROR_ALREADY_MAPPED)) {
                        const char *err_str = NULL;
                        cuGetErrorString(res, &err_str);
                        // older CUDA driver versions seem to return CUDA_ERROR_ALREADY_MAPPED
                        gds_warn("page=%p size=%zu is already registered with CUDA (%d:%s)\n", (void*)page_addr, len, res, err_str);
                        cuda_registered = true;
                }
                else if (res == CUDA_ERROR_NOT_INITIALIZED) {
                        gds_err("CUDA driver not initialized\n");
                        return EAGAIN;
                }
                else {
                        //CUCHECK(res);
                        const char *err_str = NULL;
                        cuGetErrorString(res, &err_str);
                        gds_err("Error %d (%s) while register address=%p size=%zu (original size %zu) flags=%08x\n",
                                res, err_str, (void*)page_addr, len, size, flags);
                        // TODO: handle ENOPERM
                        return EINVAL;
                }
                CUCHECK(cuMemHostGetDevicePointer(&page_dev_ptr, (void *)page_addr, 0));
        }
        else {
                // page_addr is a UVA ptr, i.e. a good device ptr already
                page_dev_ptr = (CUdeviceptr)page_addr;
        }
        gds_dbg("page_ptr=%lx page_dev_ptr=%lx\n", page_addr, (unsigned long)page_dev_ptr);

        if (dev_ptr)
                *dev_ptr = page_dev_ptr + page_off;

        // add to rangeset
        {
                range r(page_addr, page_addr+len-1);
                range_set::insert_result res = rset.insert(r);
                if (!res.second) {
                        range r = *res.first;
                        gds_dbg("range overlaps with existing\n");
                        if (!cuda_registered) {
                                gds_err("overlapping range not tracked by CUDA\n");
                                return EEXIST;
                        }
                }
        }

        // store page dev_ptr
        pinned_ranges[page_addr] = page_dev_ptr;

        return 0;
}

//-----------------------------------------------------------------------------
// TODO: !!!
int gds_unregister_mem(void *ptr, size_t size)
{
        gds_dbg("ptr=%p size=%zu\n", ptr, size);
        // r = rangeset.find()
        // if not_found
        //   return EINVAL;
        // if overlapping with other ranges
        //   return EINVAL;
        // if fully_contained
        //   remove from rangeset
        return 0;
}

