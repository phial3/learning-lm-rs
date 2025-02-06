#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template <typename T>
__device__ void matmul_transb(
    T* C,
    T beta,
    T* A,
    T* B,
    T alpha,
    int M,
    int N,
    int P
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        T sum = T(0.0f);
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[col * N + k];
        }
        C[row * P + col] = beta * C[row * P + col] + alpha * sum;
    }
}

#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void matmul_transb_##RUST_NAME( \
        TYPENAME* C, \
        TYPENAME beta, \
        TYPENAME* A, \
        TYPENAME* B, \
        TYPENAME alpha, \
        int M, \
        int N, \
        int P \
    ) { \
        matmul_transb<TYPENAME>(C, beta, A, B, alpha, M, N, P); \
    } \

#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)