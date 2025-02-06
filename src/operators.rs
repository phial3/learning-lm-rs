use std::iter::Sum;

use num_traits::Float;

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather<T: Copy + Default>(y: &mut Tensor<T>, indices: &Tensor<u32>, table: &Tensor<T>) {
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
pub fn rope<T: Copy + Default + Float>(y: &mut Tensor<T>, start_pos: usize, theta: T) {
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
                let freq = T::from(pos).unwrap()
                    / theta.powf(T::from(i * 2).unwrap() / T::from(d).unwrap());
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax<T: Copy + Default + Float + Sum>(y: &mut Tensor<T>) {
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
                .sum::<T>();

            (0..boundary).for_each(|j| data[offset + j] = data[offset + j] / sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = T::zero());
        }
    }
}

pub fn rms_norm<T: Copy + Default + Float + Sum>(
    y: &mut Tensor<T>,
    x: &Tensor<T>,
    w: &Tensor<T>,
    epsilon: T,
) {
    let len = y.size();
    assert!(len == x.size());
    let shape = y.shape().clone();
    let y_data = unsafe { y.data_mut() };
    let x = x.data();
    let w = w.data();
    for i in 0..shape[0] {
        let mut sum = T::zero();
        for j in 0..shape[1] {
            sum = sum + x[i * shape[1] + j] * x[i * shape[1] + j];
        }
        let rms = (sum / T::from(shape[1]).unwrap() + epsilon).sqrt();
        for j in 0..shape[1] {
            let idx = i * shape[1] + j;
            y_data[idx] = w[j] * (x[idx] / rms);
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu<T: Copy + Default + Float + Sum>(y: &mut Tensor<T>, x: &Tensor<T>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    let _x = _x
        .iter()
        .map(|the_x| *the_x * (T::one() / (T::one() + (-*the_x).exp())))
        .collect::<Vec<T>>();
    let mut idx = 0;
    for the_y in _y.iter_mut() {
        *the_y = *the_y * _x[idx];
        idx += 1;
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb<T: Copy + Default + Float>(
    c: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: T,
) {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let p = b.shape()[0];
    let c_data = unsafe { c.data_mut() };
    let a = a.data();
    let b = b.data();
    for i in 0..m {
        for j in 0..p {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + a[i * n + k] * b[j * n + k];
            }
            c_data[i * p + j] = beta * c_data[i * p + j] + alpha * sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot<T: Copy + Default + Float>(x: &Tensor<T>, y: &Tensor<T>) -> T {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = T::zero();
    for i in 0..len {
        sum = sum + x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample<T: Copy + Default + Float>(
    x: &Tensor<T>,
    top_p: T,
    top_k: u32,
    temperature: T,
) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= T::zero() || top_k < 2 || top_p <= T::zero() {
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
    impl<T: Float> From<(usize, &T)> for Probability {
        #[inline]
        fn from((i, p): (usize, &T)) -> Self {
            Self {
                val: p.to_f32().unwrap(),
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
        logits[i].val =
            logits[i - 1].val + ((logits[i].val - max) / temperature.to_f32().unwrap()).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p.to_f32().unwrap();
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
pub fn test_mulmat_transb2() {
    let a = Tensor::<f32>::new(
        vec![
            0.9999995, 0.9999995, 0.9999995, 0.9999995, 0.9999995, 0.9999995, 0.9999995, 0.9999995,
        ],
        &vec![4, 2],
    );
    let b = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![3, 2]);
    let mut c = Tensor::<f32>::default(&vec![4, 3]);

    matmul_transb(&mut c, 0., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(
            vec![
                0.29999986, 0.6999997, 1.0999994, 0.29999986, 0.6999997, 1.0999994, 0.29999986,
                0.6999997, 1.0999994, 0.29999986, 0.6999997, 1.0999994
            ],
            &vec![4, 3]
        ),
        1e-3
    ));
}
