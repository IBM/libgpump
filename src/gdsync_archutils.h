// Copyright (C) 2018 IBM Corporation. All rights reserved.
// $COPYRIGHT$

#pragma once

static inline void arch_cpu_relax(void) ;
static void mb(void) ;
static void rmb(void) ;
static void wmb(void) ;
#if defined(__x86_64__) || defined (__i386__)

#include "i386/cpufunc.h"
static inline void arch_cpu_relax(void)
{
        ia32_pause() ;
}

static void mb(void)
{
   mfence() ;
}
static void rmb(void)
{
   lfence() ;
}
static void wmb(void)
{
   sfence() ;
}
#elif defined(__powerpc__)

#include "powerpc/cpufunc.h"
static void arch_cpu_relax(void)
{
}

static void wmb(void)
{
        powerpc_sync() ;
}
static void rmb(void)
{
        powerpc_sync() ;
}

#else
#error "platform not supported"
#endif
