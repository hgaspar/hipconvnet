#include "hip_runtime.h"
/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <nvmatrix_kernels.cuh>

__global__ void kTile(hipLaunchParm lp,const float* src, float* tgt, const uint srcWidth, const uint srcHeight, const uint tgtWidth, const uint tgtHeight) {
    const int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int numThreads = hipBlockDim_x * hipGridDim_x;
    //    const unsigned int numEls = tgtWidth * tgtHeight;
    for (uint i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
        const uint y = i / tgtWidth;
        const uint x = i % tgtWidth;
        const uint srcY = y % srcHeight;
        const uint srcX = x % srcWidth;
        tgt[i] = src[srcY * srcWidth + srcX];
    }
}

__global__ void kDotProduct_r(hipLaunchParm lp, float* a, float* b, float* target, const uint numCols, const uint numElements) {
    __shared__ float shmem[DP_BLOCKSIZE];

    uint eidx = DP_BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    shmem[hipThreadIdx_x] = 0;
    if (eidx < numCols) {
        for (; eidx < numElements; eidx += numCols) {
            shmem[hipThreadIdx_x] += a[eidx] * b[eidx];
        }
    }
    __syncthreads();
    if (hipThreadIdx_x < 256) {
        shmem[hipThreadIdx_x] += shmem[hipThreadIdx_x + 256];
    }
    __syncthreads();
    if (hipThreadIdx_x < 128) {
        shmem[hipThreadIdx_x] += shmem[hipThreadIdx_x + 128];
    }
    __syncthreads();
    if (hipThreadIdx_x < 64) {
        shmem[hipThreadIdx_x] += shmem[hipThreadIdx_x + 64];
    }
    __syncthreads();
    if (hipThreadIdx_x < 32) {
        volatile float* mysh = &shmem[hipThreadIdx_x];
        *mysh += mysh[32];
        *mysh += mysh[16];
        *mysh += mysh[8];
        *mysh += mysh[4];
        *mysh += mysh[2];
        *mysh += mysh[1];
        if (hipThreadIdx_x == 0) {
            target[hipBlockIdx_x] = *mysh;
        }
    }
}

__global__ void kSetupCurand(hipLaunchParm lp, curandState *state, unsigned long long seed) {
    const uint tidx = NUM_RND_THREADS_PER_BLOCK * hipBlockIdx_x + hipThreadIdx_x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, tidx, 0, &state[tidx]);
}

