#include <cuda_bf16.h>
#include <cuda_fp16.h>

template<typename T>
__device__ T maximum(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
__device__ T zero() {
    return T(0.0f);
}

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
__device__ void masked_softmax(
    T* data,
    const int batch,
    const int seq_len,
    const int total_seq_len
) {
    int b = blockIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b < batch && i < seq_len) {
        int base = b * seq_len * total_seq_len;
        int offset = base + i * total_seq_len;
        int boundary = total_seq_len - seq_len + i + 1;
        
        T max_val = data[offset];
        for (int j = 1; j < boundary; j++) {
            max_val = maximum(max_val, data[offset + j]);
        }
        
        T sum = zero<T>();
        for (int j = 0; j < boundary; j++) {
            T val = exp_wrapper(data[offset + j] - max_val);
            data[offset + j] = val;
            sum += val;
        }
        
        for (int j = 0; j < boundary; j++) {
            data[offset + j] = data[offset + j] / sum;
        }
        for (int j = boundary; j < total_seq_len; j++) {
            data[offset + j] = zero<T>();
        }
    }
}

#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void masked_softmax_##RUST_NAME( \
        TYPENAME* data, \
        const int batch, \
        const int seq_len, \
        const int total_seq_len \
    ) { \
        masked_softmax<TYPENAME>(data, batch, seq_len, total_seq_len); \
    } \

#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)