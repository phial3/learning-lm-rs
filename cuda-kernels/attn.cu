#include <cuda_bf16.h>
#include <cuda_fp16.h>

template<typename T>
__device__ T zero() {
    return T(0.0f);
}

template<typename T>
__device__ void attention_scores_kernel(
    T* scores,           // [n_kv_h, n_groups, seq_len, total_seq_len]
    const T* q,          // [seq_len, n_kv_h * n_groups * dqkv] 
    const T* k,          // [total_seq_len, n_kv_h * dqkv]
    const T scale,       // scalar
    const int seq_len,
    const int total_seq_len,
    const int n_kv_h,
    const int n_groups,
    const int dqkv
) {
    const int kv_head = blockIdx.x;
    const int group = blockIdx.y;
    const int q_seq = blockIdx.z;
    const int k_seq = threadIdx.x;

    if (kv_head >= n_kv_h || group >= n_groups || 
        q_seq >= seq_len || k_seq >= total_seq_len) {
        return;
    }

    T dot_product = zero<T>();
    for (int d = 0; d < dqkv; d++) {
        const int q_idx = q_seq * (n_kv_h * n_groups * dqkv) 
            + (kv_head * n_groups + group) * dqkv + d;
        const int k_idx = k_seq * (n_kv_h * dqkv) + kv_head * dqkv + d;
        dot_product += q[q_idx] * k[k_idx];
    }

    const int score_idx = kv_head * (n_groups * seq_len * total_seq_len)
        + group * (seq_len * total_seq_len)
        + q_seq * total_seq_len
        + k_seq;
    scores[score_idx] = dot_product * scale;
}

template<typename T>
__device__ void attention_output_kernel(
    T* output,           // [seq_len, n_kv_h * n_groups * dqkv]
    const T* scores,     // [n_kv_h, n_groups, seq_len, total_seq_len]
    const T* v,          // [total_seq_len, n_kv_h * dqkv]
    const int seq_len,
    const int total_seq_len, 
    const int n_kv_h,
    const int n_groups,
    const int dqkv
) {
    const int kv_head = blockIdx.x;
    const int group = blockIdx.y;
    const int q_seq = blockIdx.z;
    const int d = threadIdx.x;

    if (kv_head >= n_kv_h || group >= n_groups || 
        q_seq >= seq_len || d >= dqkv) {
        return;
    }

    T sum = zero<T>();
    for (int k_seq = 0; k_seq < total_seq_len; k_seq++) {
        const int score_idx = kv_head * (n_groups * seq_len * total_seq_len)
            + group * (seq_len * total_seq_len)
            + q_seq * total_seq_len
            + k_seq;
        const int v_idx = k_seq * (n_kv_h * dqkv) + kv_head * dqkv + d;
        sum += scores[score_idx] * v[v_idx];
    }

    const int out_idx = q_seq * (n_kv_h * n_groups * dqkv)
        + (kv_head * n_groups + group) * dqkv + d;
    output[out_idx] = sum;
}

#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void attention_scores_##RUST_NAME( \
        TYPENAME* scores, \
        const TYPENAME* q, \
        const TYPENAME* k, \
        const TYPENAME scale, \
        const int seq_len, \
        const int total_seq_len, \
        const int n_kv_h, \
        const int n_groups, \
        const int dqkv \
    ) { \
        attention_scores_kernel<TYPENAME>( \
            scores, q, k, scale, seq_len, total_seq_len, \
            n_kv_h, n_groups, dqkv); \
    } \
    extern "C" __global__ void attention_output_##RUST_NAME( \
        TYPENAME* output, \
        const TYPENAME* scores, \
        const TYPENAME* v, \
        const int seq_len, \
        const int total_seq_len, \
        const int n_kv_h, \
        const int n_groups, \
        const int dqkv \
    ) { \
        attention_output_kernel<TYPENAME>( \
            output, scores, v, seq_len, total_seq_len, \
            n_kv_h, n_groups, dqkv); \
    }

#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)
