use crate::tensor::Tensor;
use core::panic;
use rayon::prelude::*;

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

// shape boardcast
pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    let mut shape = Vec::with_capacity(max_len);
    // 逐个从后往前匹配维度
    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        // 维度要么相等，要么有一个是1，否则不能广播
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            panic!("Cannot broadcast shapes {:?} and {:?}", shape1, shape2);
        }
        shape[max_len - 1 - i] = dim1.max(dim2);
    }
    shape
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let shape_x = x.shape();
    let shape_y = y.shape();

    // 检查 x 和 y 的最后一维是否相同
    let n_x = *shape_x.last().unwrap();
    let n_y = *shape_y.last().unwrap();
    assert_eq!(n_x, n_y, "x and y must have the same last dimension");

    // 检查 w 的长度是否与最后一维 n 相同
    assert_eq!(
        w.size(),
        n_x,
        "w must have the same length as the last dimension of x"
    );

    // 计算 x 和 y 的总元素数
    let len_x = x.size();
    let len_y = y.size();

    // 计算 x 和 y 的最后一维之前的维度大小
    let num_vectors_x = len_x / n_x;
    let num_vectors_y = len_y / n_y;

    // 检查 x 和 y 的向量数量是否匹配
    assert_eq!(
        num_vectors_x, num_vectors_y,
        "x and y must have the same number of vectors"
    );

    // 获取 y 的可变数据引用
    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();
    let w_data = w.data();

    // 对每个向量进行 RMS 归一化
    for i in 0..num_vectors_x {
        let start_index_x = i * n_x;
        let end_index_x = start_index_x + n_x;

        let start_index_y = i * n_y;

        // 计算 RMS
        let rms: f32 = x_data[start_index_x..end_index_x]
            .iter()
            .map(|&x| x * x)
            .sum();
        let mu = (rms / (n_x as f32) + epsilon).sqrt();

        // 计算归一化结果
        for j in 0..n_x {
            y_data[start_index_y + j] = x_data[start_index_x + j] * w_data[j] / mu;
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    assert!(
        y.shape() == x.shape(),
        "swiglu: x and y must have the same shape"
    );

    let len = y.size();
    let y = unsafe { y.data_mut() };
    let x = x.data();

    for i in 0..len {
        y[i] = y[i] * x[i] * (1.0 / ((-x[i]).exp() + 1.0));
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let shape_c = c.shape();
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert!(shape_b.len() == 2 && shape_a.len() == 2 && shape_c.len() == 2);

    let m = shape_a[0];
    let k_dim = shape_a[1];
    let n = shape_b[0];
    assert_eq!(
        shape_a[1], shape_b[1],
        "a cols {} != b cols {}",
        shape_a[1], shape_b[1]
    );
    assert_eq!(shape_c, &[m, n], "c shape mismatch");

    // 获取底层数据指针
    let c_data = unsafe { c.data_mut() };
    let a_data = a.data();
    let b_data = b.data();

    // 并行处理每一行
    c_data.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        let a_row = &a_data[i * k_dim..(i + 1) * k_dim];

        // 对每列进行并行计算
        c_row.par_iter_mut().enumerate().for_each(|(j, c_val)| {
            let mut sum = 0.0;
            let b_row = &b_data[j * k_dim..(j + 1) * k_dim];

            // 展开循环以提高性能
            for k in (0..k_dim).step_by(4) {
                let end = (k + 4).min(k_dim);
                sum += a_row[k..end]
                    .iter()
                    .zip(&b_row[k..end])
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>();
            }

            *c_val = *c_val * beta + sum * alpha;
        });
    });
}

// 辅助函数保持与之前相同的实现（移除了显式转置）
#[allow(unused)]
fn caculate2mat(a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) -> Vec<f32> {
    let shape_a = a.shape();
    let shape_b = b.shape();
    let _m = shape_a[0];
    let n = shape_b[0];
    let k_dim = shape_a[1];

    a.data()
        .par_chunks(shape_a[1])
        .flat_map(|a_row| {
            (0..n)
                .into_par_iter()
                .map(|j| {
                    let b_row = &b.data()[j * k_dim..(j + 1) * k_dim];
                    a_row.iter().zip(b_row).map(|(&a, &b)| a * b).sum::<f32>() * alpha
                })
                .collect::<Vec<_>>()
        })
        .collect()
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
                val: *p,
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

#[test]
#[should_panic]
fn test_invalid_broadcast() {
    broadcast_shapes(&[2, 3], &[3, 2]);
}
