use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // 1. 检查输入的合法性
    assert_eq!(x.shape(), y.shape(), "Input and output tensors must have the same shape");
    let x_shape = x.shape();
    assert!(!x_shape.is_empty(), "Input tensor cannot be empty");

    // 获取最后一维的大小
    let last_dim = *x_shape.last().unwrap();

    // 检查权重向量维度
    assert_eq!(w.shape(), &vec![last_dim], "Weight tensor must match the last dimension");

    // 2. 获取数据访问
    let x_data = x.data();
    let w_data = w.data();
    let y_data = unsafe { y.data_mut() };

    // 计算向量数量（总元素数除以最后一维的大小）
    let total_size = x.size();
    let num_vectors = total_size / last_dim;

    // 3. 对每个向量进行 RMS Normalization
    for i in 0..num_vectors {
        let start_idx = i * last_dim;

        // 计算均方和
        let mut sum_squares = 0.0f32;
        for j in 0..last_dim {
            let val = x_data[start_idx + j];
            sum_squares += val * val;
        }

        // 计算 RMS (root mean square)
        let rms = (sum_squares / last_dim as f32 + epsilon).sqrt();

        // 应用归一化和权重
        for j in 0..last_dim {
            let idx = start_idx + j;
            y_data[idx] = (x_data[idx] / rms) * w_data[j];
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size(), "Tensors must have the same size");

    // Get mutable slice for y and immutable slice for x
    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();

    // Perform element-wise SwiGLU operation
    // SwiGLU: y = silu(x) * y
    // where silu(x) = x * sigmoid(x)
    for i in 0..len {
        // Calculate sigmoid(x): 1 / (1 + e^(-x))
        let sigmoid_x = 1.0 / (1.0 + (-x_data[i]).exp());

        // Calculate silu(x) = x * sigmoid(x)
        let silu_x = x_data[i] * sigmoid_x;

        // Multiply result with y (in-place)
        y_data[i] *= silu_x;
    }
}

/// 矩阵乘（Transpose B）算子
/// C = beta * C + alpha * A @ B^T
/// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 1. 获取维度信息并进行检查
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();

    assert_eq!(a_shape.len(), 2, "Matrix A must be 2-dimensional");
    assert_eq!(b_shape.len(), 2, "Matrix B must be 2-dimensional");
    assert_eq!(c_shape.len(), 2, "Matrix C must be 2-dimensional");

    let (m, k) = (a_shape[0], a_shape[1]);   // A: m × k
    let (n, b_k) = (b_shape[0], b_shape[1]); // B: n × b_k

    // 修改维度检查：A的列数(k)应该等于B的列数(b_k)
    assert_eq!(k, b_k, "Inner dimensions must match for A @ B^T: A is {}×{}, B is {}×{}", m, k, n, b_k);
    assert_eq!(c_shape[0], m, "Output matrix C must have {} rows", m);
    assert_eq!(c_shape[1], n, "Output matrix C must have {} columns", n);

    // 2. 获取数据访问
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    // 3. 实现 C = beta * C + alpha * A @ B^T
    // 对每个输出元素 C[i,j] 计算
    for i in 0..m {
        for j in 0..n {
            let c_idx = i * n + j;

            // 首先应用 beta * C
            let mut sum = beta * c_data[c_idx];

            // 计算 A @ B^T 的对应元素
            // C[i,j] = sum(A[i,k] * B[j,k]) 因为B是转置的
            for p in 0..k {
                let a_idx = i * k + p;        // A[i,p]
                let b_idx = j * b_k + p;      // B[j,p]
                sum += alpha * a_data[a_idx] * b_data[b_idx];
            }

            c_data[c_idx] = sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
#[warn(dead_code)]
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
