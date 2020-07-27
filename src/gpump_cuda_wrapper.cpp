// Copyright (C) IBM Corporation 2018. All Rights Reserved
//
//    This program is licensed under the terms of the Eclipse Public License
//    v1.0 as published by the Eclipse Foundation and available at
//    http://www.eclipse.org/legal/epl-v10.html
//
//    
//    
// $COPYRIGHT$

#include <dlfcn.h>
#include <stdio.h>

//Disable wrappers so that preprocessor can expand
//CUDA_SYMBOL_STRING to the correct string
#define __GPUMP_WRAPPER_DISABLE
#include "gpump_cuda_wrapper.h"

struct gpump_cuda_wrapper gpump_wrapper = {0};

static inline void *gpump_dlsym(void *handle, const char *symname)
{
    char *error;
    void *sym;

    sym = dlsym(handle, symname);
    if ((error = dlerror()) != NULL)  {
        fprintf(stderr, "Error looking up symbol %s : %s\n", symname, error);
        dlclose(handle);
        return NULL;
    }
    return sym;
}

int gpump_cuda_wrapper_init()
{
    void *handle;
//    fprintf(stderr,"gpump_cuda_wrapper_init\n");
    handle = dlopen("libcuda.so", RTLD_NOW|RTLD_GLOBAL);
    if(!handle) {
        fprintf(stderr, "Error loading cuda library: %s\n", dlerror());
        return 1;
    }
    /*Clear any existing error*/
    dlerror();

    do {
        if(!(gpump_wrapper._cuCtxGetCurrent = (CUresult (*) (CUcontext*))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuCtxGetCurrent)))) break;
        if(!(gpump_wrapper._cuCtxGetDevice = (CUresult (*) (CUdevice*))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuCtxGetDevice)))) break;
        if(!(gpump_wrapper._cuCtxPopCurrent = (CUresult (*) (CUcontext*))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuCtxPopCurrent)))) break;
        if(!(gpump_wrapper._cuCtxPushCurrent = (CUresult (*) (CUcontext))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuCtxPushCurrent)))) break;
        if(!(gpump_wrapper._cuDeviceGet = (CUresult (*) (CUdevice*, int))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuDeviceGet)))) break;
        if(!(gpump_wrapper._cuDeviceGetAttribute = (CUresult (*) (int*, CUdevice_attribute, CUdevice))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuDeviceGetAttribute)))) break;
        if(!(gpump_wrapper._cuDeviceGetCount = (CUresult (*) (int*))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuDeviceGetCount)))) break;
        if(!(gpump_wrapper._cuDevicePrimaryCtxRelease = (CUresult (*) (CUdevice))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuDevicePrimaryCtxRelease)))) break;
        if(!(gpump_wrapper._cuDevicePrimaryCtxRetain = (CUresult (*) (CUcontext*, CUdevice))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuDevicePrimaryCtxRetain)))) break;
        if(!(gpump_wrapper._cuGetErrorName = (CUresult (*) (CUresult, const char**))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuGetErrorName)))) break;
        if(!(gpump_wrapper._cuGetErrorString = (CUresult (*) (CUresult, const char**))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuGetErrorString)))) break;
        if(!(gpump_wrapper._cuInit = (CUresult (*) (unsigned int))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuInit)))) break;
        if(!(gpump_wrapper._cuMemGetAddressRange = (CUresult (*) (CUdeviceptr*, size_t*, CUdeviceptr))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuMemGetAddressRange)))) break;
        if(!(gpump_wrapper._cuMemHostGetDevicePointer = (CUresult (*) (CUdeviceptr*, void*, unsigned int))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuMemHostGetDevicePointer)))) break;
        if(!(gpump_wrapper._cuMemHostRegister = (CUresult (*) (void*, size_t, unsigned int))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuMemHostRegister)))) break;
        if(!(gpump_wrapper._cuPointerGetAttribute = (CUresult (*) (void*, CUpointer_attribute, CUdeviceptr))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuPointerGetAttribute)))) break;
        if(!(gpump_wrapper._cuPointerSetAttribute = (CUresult (*) (const void*, CUpointer_attribute, CUdeviceptr))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuPointerSetAttribute)))) break;
        if(!(gpump_wrapper._cuStreamBatchMemOp = (CUresult (*) (CUstream, unsigned int, CUstreamBatchMemOpParams*, unsigned int))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuStreamBatchMemOp)))) break;
        if(!(gpump_wrapper._cuStreamGetCtx = (CUresult (*) (CUstream, CUcontext*))
           gpump_dlsym(handle, CUDA_SYMBOL_STRING(cuStreamGetCtx)))) break;

        return 0;
    } while(0);

    return 1;
}
