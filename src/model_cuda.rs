use std::fs::File;
use std::iter::Sum;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators_cuda::{self as OP};
use crate::params::{FromBytes, LLamaParams};
use crate::tensor::Tensor;
use num_traits::Float;
use safetensors::SafeTensors;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: T,                 // epsilon for RMS normalization
    rope_theta: T,          // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
    pub operator: OP::CudaOperator,
}

impl<T: Copy + Default + FromBytes + Float + Sum + OP::CudaDType> Llama<T> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: T::from(config.rms_norm_eps).unwrap(),
            rope_theta: T::from(config.rope_theta).unwrap(),
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
            #[cfg(feature = "cuda")]
            operator: OP::CudaOperator::new(),
        }
    }

    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        self.operator
            .gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            self.operator.rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            self.operator.matmul_transb(
                q,
                T::zero(),
                &hidden_states,
                &self.params.wq[layer],
                T::one(),
            );
            self.operator.matmul_transb(
                k,
                T::zero(),
                &hidden_states,
                &self.params.wk[layer],
                T::one(),
            );
            self.operator.matmul_transb(
                v,
                T::zero(),
                &hidden_states,
                &self.params.wv[layer],
                T::one(),
            );

            self.operator.rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            self.operator.rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self.operator.self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            self.operator.matmul_transb(
                &mut residual,
                T::one(),
                &hidden_states,
                &self.params.wo[layer],
                T::one(),
            );

            mlp(
                &self.operator,
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<T>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        self.operator.rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        self.operator.matmul_transb(
            &mut logits,
            T::zero(),
            &hidden_states,
            &self.params.lm_head,
            T::one(),
        );

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = vec![self.bos_token_id];
        result.extend_from_slice(token_ids);
        let mut cache = self.new_cache();

        let input_tensor = Tensor::new(result.clone(), &vec![result.len()]);
        let logits = self.forward(&input_tensor, &mut cache);

        let mut next_token = self
            .operator
            .random_sample(&logits, top_p, top_k, temperature);
        result.push(next_token);

        while result.len() < max_len {
            let input_tensor = Tensor::new(vec![next_token], &vec![1]);
            let logits = self.forward(&input_tensor, &mut cache);

            next_token = self
                .operator
                .random_sample(&logits, top_p, top_k, temperature);

            if next_token == self.eos_token_id {
                break;
            }

            result.push(next_token);
        }

        result
    }

    pub fn chat(
        &self,
        promps: &str,
        cache: &mut KVCache<T>,
        tokenizer: &Tokenizer,
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> String {
        let encoding = tokenizer.encode(promps, false).unwrap();
        let token_ids = encoding.get_ids();

        let input_tensor = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()]);
        let logits = self.forward(&input_tensor, cache);

        let mut next_token = self
            .operator
            .random_sample(&logits, top_p, top_k, temperature);
        let mut result = vec![next_token];

        while result.len() < max_len {
            let input_tensor = Tensor::new(vec![next_token], &vec![1]);
            let logits = self.forward(&input_tensor, cache);

            next_token = self
                .operator
                .random_sample(&logits, top_p, top_k, temperature);

            if next_token == self.eos_token_id {
                break;
            }

            result.push(next_token);
        }

        tokenizer.decode(&result, true).unwrap()
    }

    #[cfg(feature = "web")]
    pub fn chat_stream<'a>(
        &'a self,
        promps: &str,
        cache: &'a mut KVCache<T>,
        tokenizer: &'a Tokenizer,
        max_len: usize,
        top_p: T,
        top_k: u32,
        temperature: T,
    ) -> impl Stream<Item = String> + 'a {
        use async_stream::stream;

        let encoding = tokenizer.encode(promps, false).unwrap();
        let token_ids = encoding.get_ids();

        let input_tensor = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()]);
        let logits = self.forward(&input_tensor, cache);

        let mut next_token = self
            .operator
            .random_sample(&logits, top_p, top_k, temperature);
        let mut cnt = 1;
        stream! {
            let msg = tokenizer.decode(&vec![next_token], true).unwrap();

            yield msg;

            while cnt < max_len {
                let input_tensor = Tensor::new(vec![next_token], &vec![1]);
                let logits = self.forward(&input_tensor, cache);

                next_token = self.operator.random_sample(&logits, top_p, top_k, temperature);

                if next_token == self.eos_token_id {
                    break;
                }

                yield tokenizer.decode(&vec![next_token], true).unwrap();
                cnt+=1;
            }
        }
    }
}

fn mlp<T: Copy + Default + Float + Sum + OP::CudaDType>(
    op: &OP::CudaOperator,
    residual: &mut Tensor<T>,
    hidden_states: &mut Tensor<T>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: T,
) {
    op.rms_norm(hidden_states, residual, rms_w, eps);
    op.matmul_transb(gate, T::zero(), hidden_states, w_gate, T::one());
    op.matmul_transb(up, T::zero(), hidden_states, w_up, T::one());
    op.swiglu(up, gate);
    op.matmul_transb(residual, T::one(), up, w_down, T::one());
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    let op = OP::CudaOperator::new();
    mlp(
        &op,
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
