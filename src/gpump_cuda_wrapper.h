// Copyright (C) IBM Corporation 2018. All Rights Reserved
//
//    This program is licensed under the terms of the Eclipse Public License
//    v1.0 as published by the Eclipse Foundation and available at
//    http://www.eclipse.org/legal/epl-v10.html
//
//    
//    
// $COPYRIGHT$

#ifndef __GPUMP_CUDA_INTERCEPT__
#define __GPUMP_CUDA_INTERCEPT__
#include <cuda.h>

struct gpump_cuda_wrapper
{
    CUresult (*_cuCtxGetCurrent) (CUcontext* pctx);
    CUresult (*_cuCtxGetDevice) (CUdevice* device);
    CUresult (*_cuCtxPopCurrent) (CUcontext* pctx);
    CUresult (*_cuCtxPushCurrent) (CUcontext ctx);
    CUresult (*_cuDeviceGet) (CUdevice* device, int  ordinal);
    CUresult (*_cuDeviceGetAttribute) (int* pi, CUdevice_attribute attrib, CUdevice dev);
    CUresult (*_cuDeviceGetCount) (int* count);
    CUresult (*_cuDevicePrimaryCtxRelease) (CUdevice dev);
    CUresult (*_cuDevicePrimaryCtxRetain) (CUcontext* pctx, CUdevice dev);
    CUresult (*_cuGetErrorName) (CUresult error, const char** pstr);
    CUresult (*_cuGetErrorString) (CUresult error, const char** pstr);
    CUresult (*_cuInit) (unsigned int flags);
    CUresult (*_cuMemGetAddressRange) (CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
    CUresult (*_cuMemHostGetDevicePointer) (CUdeviceptr* pdptr, void* p, unsigned int flags);
    CUresult (*_cuMemHostRegister) (void* p, size_t bytesize, unsigned int flags);
    CUresult (*_cuPointerGetAttribute) (void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
    CUresult (*_cuPointerSetAttribute) (const void* value, CUpointer_attribute attribute, CUdeviceptr ptr);
    CUresult (*_cuStreamBatchMemOp) (CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramarray, unsigned int flags);
    CUresult (*_cuStreamGetCtx) (CUstream hstream, CUcontext* pctx);
};

extern struct gpump_cuda_wrapper gpump_wrapper;
int gpump_cuda_wrapper_init();

// We need to give the pre-processor a chance to replace a function, such as:
// cuMemAlloc => cuMemAlloc_v2
#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

#ifndef __GPUMP_WRAPPER_DISABLE

#undef cuCtxGetCurrent
#define cuCtxGetCurrent(pctx) gpump_wrapper._cuCtxGetCurrent(pctx)

#undef cuCtxGetDevice
#define cuCtxGetDevice(device) gpump_wrapper._cuCtxGetDevice(device)

#undef cuCtxPopCurrent
#define cuCtxPopCurrent(pctx) gpump_wrapper._cuCtxPopCurrent(pctx)

#undef cuCtxPushCurrent
#define cuCtxPushCurrent(ctx) gpump_wrapper._cuCtxPushCurrent(ctx)

#undef cuDeviceGet
#define cuDeviceGet(device, ordinal) gpump_wrapper._cuDeviceGet(device, ordinal)

#undef cuDeviceGetAttribute
#define cuDeviceGetAttribute(pi, attrib, dev) gpump_wrapper._cuDeviceGetAttribute(pi, attrib, dev)

#undef cuDeviceGetCount
#define cuDeviceGetCount(count) gpump_wrapper._cuDeviceGetCount(count)


#undef cuDevicePrimaryCtxRelease
#define cuDevicePrimaryCtxRelease(dev) gpump_wrapper._cuDevicePrimaryCtxRelease(dev)

#undef cuDevicePrimaryCtxRetain
#define cuDevicePrimaryCtxRetain(pctx, dev) gpump_wrapper._cuDevicePrimaryCtxRetain(pctx, dev)

#undef cuGetErrorName
#define cuGetErrorName(error, pstr) gpump_wrapper._cuGetErrorName(error, pstr)

#undef cuGetErrorString
#define cuGetErrorString(error, pstr) gpump_wrapper._cuGetErrorString(error, pstr)

#undef cuInit
#define cuInit(dev) gpump_wrapper._cuInit(dev)

#undef cuMemGetAddressRange
#define cuMemGetAddressRange(pbase, psize, dptr) gpump_wrapper._cuMemGetAddressRange(pbase, psize, dptr)

#undef cuMemHostGetDevicePointer
#define cuMemHostGetDevicePointer(pdptr, p, flags) gpump_wrapper._cuMemHostGetDevicePointer(pdptr, p, flags)

#undef cuMemHostRegister
#define cuMemHostRegister(p, bytesize, flags) gpump_wrapper._cuMemHostRegister(p, bytesize, flags)

#undef cuPointerGetAttribute
#define cuPointerGetAttribute(data, attribute, ptr) gpump_wrapper._cuPointerGetAttribute(data, attribute, ptr)

#undef cuPointerSetAttribute
#define cuPointerSetAttribute(value, attribute, ptr) gpump_wrapper._cuPointerSetAttribute(value, attribute, ptr)

#undef cuStreamBatchMemOp
#define cuStreamBatchMemOp(stream, count, paramarray, flags) gpump_wrapper._cuStreamBatchMemOp(stream, count, paramarray, flags)

#undef cuStreamGetCtx
#define cuStreamGetCtx(hstream, pctx) gpump_wrapper._cuStreamGetCtx(hstream, pctx)

#endif //__GPUMP_WRAPPER_DISABLE

#endif
