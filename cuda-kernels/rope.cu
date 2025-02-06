#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template <typename T>
__device__ void rope(
    T* data,
    int start_pos,
    float theta,
    int seq_len,
    int n_heads,
    int d
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    int head = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (tok < seq_len && head < n_heads && i < d/2) {
        int pos = start_pos + tok;
        int base_idx = tok * n_heads * d + head * d;
        
        T a = data[base_idx + i];
        T b = data[base_idx + i + d/2];
        
        float freq = (float)pos / powf(theta, (2.0f * i) / (float)d);
        float sin_val, cos_val;
        sincosf(freq, &sin_val, &cos_val);
        
        data[base_idx + i] = a * T(cos_val) - b * T(sin_val);
        data[base_idx + i + d/2] = b * T(cos_val) + a * T(sin_val);
    }
}

#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void rope_##RUST_NAME( \
        TYPENAME* data, \
        int start_pos, \
        float theta, \
        int seq_len, \
        int n_heads, \
        int d \
    ) { \
        rope<TYPENAME>(data, start_pos, theta, seq_len, n_heads, d); \
    } \


#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)