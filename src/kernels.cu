// kernels.cu
extern "C" {

    // gather_kernel: 每个线程处理一个token，从表中收集数据
    __global__ void gather_kernel(const float* table, const unsigned int* indices, float* y, unsigned int dim, unsigned int n) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n) {
            unsigned int tableIndex = indices[idx];
            for (unsigned int i = 0; i < dim; i++) {
                y[idx * dim + i] = table[tableIndex * dim + i];
            }
        }
    }

    // rope_kernel: 每个线程处理一个元素，实现旋转位置编码（RoPE）
    __global__ void rope_kernel(float* y, unsigned int start_pos, float theta, unsigned int seq_len, unsigned int n_heads, unsigned int d) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int total = seq_len * n_heads * d;
        if(tid < total) {
            // 计算 token、head 和向量内位置
            unsigned int tok = tid / (n_heads * d);
            unsigned int rem = tid % (n_heads * d);
            unsigned int head = rem / d;
            unsigned int pos_in_head = rem % d;
            // 对前半部分进行旋转变换
            if(pos_in_head < d/2) {
                unsigned int pos = start_pos + tok;
                float a = y[tid];
                float b = y[tid + d/2]; // 假定连续存储：前半和后半各占 d/2 长
                float freq = pos / powf(theta, (float)(pos_in_head * 2) / d);
                float s = sinf(freq);
                float c = cosf(freq);
                y[tid] = a * c - b * s;
                y[tid + d/2] = b * c + a * s;
            }
        }
    }

    // masked_softmax_kernel: 对每行进行带掩码的softmax操作
    __global__ void masked_softmax_kernel(float* y, unsigned int batch, unsigned int seq_len, unsigned int total_seq_len) {
        // 每个 block 处理一个batch
        // 每个 thread 处理一个 其中一行kv结果
        unsigned int b = blockIdx.x;
        unsigned int i = threadIdx.x;
        if(b < batch && i < seq_len) {
            unsigned int offset = b * seq_len * total_seq_len + i * total_seq_len;
            unsigned int boundary = total_seq_len - seq_len + i + 1;
            // 计算最大值
            float max_val = y[offset];
            for (unsigned int j = 0; j < boundary; j++) {
                float val = y[offset + j];
                if (val > max_val) max_val = val;
            }
            float sum = 0.f;
            for (unsigned int j = 0; j < boundary; j++) {
                y[offset + j] = expf(y[offset + j] - max_val);
                sum += y[offset + j];
            }
            for (unsigned int j = 0; j < boundary; j++) {
                y[offset + j] /= sum;
            }
            for (unsigned int j = boundary; j < total_seq_len; j++) {
                y[offset + j] = 0.0f;
            }
        }
    }

    // rms_norm_kernel: 实现RMS归一化
    __global__ void rms_norm_kernel(float* y, const float* x, const float* w, unsigned int n, float epsilon, unsigned int batch) {
        unsigned int b = blockIdx.x;
        if(b < batch) {
            unsigned int offset = b * n;
            float sum_sq = 0.f;
            for (unsigned int i = 0; i < n; i++) {
                float val = x[offset + i];
                sum_sq += val * val;
            }
            float norm = sqrtf(sum_sq / n + epsilon);
            unsigned int tid = threadIdx.x;
            if(tid < n) {
                y[offset + tid] = w[tid] * x[offset + tid] / norm;
            }
        }
    }

    // swiglu_kernel: 实现SwiGLU激活函数
    __global__ void swiglu_kernel(float* y, const float* x, unsigned int len) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < len) {
            float xi = x[idx];
            float sigmoid = 1.0f / (1.0f + expf(-xi));
            float silu = xi * sigmoid;
            y[idx] *= silu;
        }
    }

    // matmul_transb_kernel: 实现矩阵乘法（B矩阵转置）
    __global__ void matmul_transb_kernel(const float* a, const float* b, float* c, unsigned int m, unsigned int n, unsigned int k, float beta, float alpha) {
        unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if(row < m && col < n) {
            float sum = 0.0f;
            for (unsigned int p = 0; p < k; p++) {
                sum += a[row * k + p] * b[col * k + p];
            }
            c[row * n + col] = beta * c[row * n + col] + alpha * sum;
        }
    }

    // matmul_transb_kernel_v2: 优化后的矩阵乘法（B矩阵转置），使用共享内存
    #define BLOCK_SIZE 32
    __global__ void matmul_transb_kernel_v2(const float* A, const float* B, float* C, unsigned int M, unsigned int N, unsigned int K, float beta, float alpha) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int idx = bx * blockDim.x + tx;
        int idy = by * blockDim.y + ty;
        if (idx >= N || idy >= M) return;

        __shared__ float sA[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ float sB[BLOCK_SIZE * BLOCK_SIZE];

        A = &A[(by * BLOCK_SIZE) * K];
        B = &B[bx * BLOCK_SIZE];
        C = &C[(by * BLOCK_SIZE) * N + bx * BLOCK_SIZE];

        float sum = 0.0f;
        for(int k = 0; k < K; k += BLOCK_SIZE) {
            // global -> shared
            sA[ty * BLOCK_SIZE + tx] = k * BLOCK_SIZE + tx < K ? A[ty * K + tx] : 0.0f;
            sB[ty * BLOCK_SIZE + tx] = k * BLOCK_SIZE + ty < K ? B[ty * N + tx] : 0.0f;
            __syncthreads();
            A = A + BLOCK_SIZE;
            B = B + BLOCK_SIZE * N;
            // calc sum
            for(int i = 0; i < BLOCK_SIZE; i++) {
                sum += sA[ty * BLOCK_SIZE + i] * sB[i * BLOCK_SIZE + tx];
            }
            __syncthreads();
        }

        C[ty * N + tx] = beta * C[ty * N + tx] + alpha * sum;
    }

} // extern "C"