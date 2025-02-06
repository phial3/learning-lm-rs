#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template <typename T>
__device__ void gather(
    T* output,
    const unsigned int* indices,
    const T* table,
    int length,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < length && dim_idx < dim) {
        unsigned int table_idx = indices[idx];
        output[idx * dim + dim_idx] = table[table_idx * dim + dim_idx];
    }
}

#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void gather_##RUST_NAME( \
        TYPENAME* output, \
        const unsigned int* indices, \
        const TYPENAME* table, \
        int length, \
        int dim \
    ) {\
    gather<TYPENAME>(output, indices, table, length, dim); \
    } \


#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)