#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template<typename T>
__device__ T exp_wrapper(T x) {
    return expf(x);
}

template<>
__device__ __half exp_wrapper(__half x) {
    return hexp(x);
}

template<>
__device__ __nv_bfloat16 exp_wrapper(__nv_bfloat16 x) {
    return hexp(x);
}

template <typename T>
__device__ void swiglu(
    T* y,
    const T* x,
    int num_elements
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const T x_val = x[i];
    const T sigmoid = T(1.0f) / (T(1.0f) + exp_wrapper(-x_val));
    const T silu = x_val * sigmoid;
    
    y[i] *= silu;
}


#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void swiglu_##RUST_NAME( \
        TYPENAME* y, \
        const TYPENAME* x, \
        int num_elements \
    ) { \
        swiglu<TYPENAME>(y, x, num_elements); \
    } \


#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)