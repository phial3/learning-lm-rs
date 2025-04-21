use crate::tensor::Tensor;
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl<T: Default + Copy + 'static> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&[max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&[max_seq_len, dim]))
                .collect(),
            max_seq_len,
            dim,
            length: init_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &[self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &[self.length - start, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize) {
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }

    // 回滚方法
    pub fn rollback(&mut self, sub_length: usize) {
        assert!(sub_length <= self.length);

        let new_length = self.length - sub_length;
        self.length = new_length;
    }
}
