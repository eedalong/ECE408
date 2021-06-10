
#include "wb.h"

#ifdef WB_USE_CUDA
typedef struct st_wbCUDAMemory_t {
    void * mem;
    size_t sz;
} wbCUDAMemory_t;

static wbCUDAMemory_t _cudaMemoryList[1024];
static int idx = 0;

size_t _cudaMallocSize = 0;

cudaError_t wbCUDAMalloc(void ** devPtr, size_t sz) {
    cudaError_t err = cudaMalloc(devPtr, sz);
    if (idx == 0) {
        memset(_cudaMemoryList, 0, sizeof(wbCUDAMemory_t) * 1024);
    }
    _cudaMallocSize += sz;
    _cudaMemoryList[idx].mem = *devPtr;
    _cudaMemoryList[idx].sz = sz;
    idx++;
    return err;
}

cudaError_t wbCUDAFree(void * mem) {
    if (idx == 0) {
        memset(_cudaMemoryList, 0, sizeof(wbCUDAMemory_t) * 1024);
    }
    for (int ii = 0; ii < idx; ii++) {
        if (_cudaMemoryList[ii].mem != NULL &&
                _cudaMemoryList[ii].mem == mem) {
            cudaError_t err = cudaFree(mem);
            _cudaMallocSize -= _cudaMemoryList[ii].sz;
            _cudaMemoryList[idx].mem = NULL;
            return err;
        }
    }
    return cudaSuccess;
}

#endif /* WB_USE_CUDA */

