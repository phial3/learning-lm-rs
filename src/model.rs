use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{
    gather, masked_softmax, matmul_transb, random_sample, rms_norm, rope, swiglu,
};
use crate::params::LLamaParams;
use crate::tensor::Tensor;

use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;
use std::vec;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    //bos_token_id: u32,      // start token id
    eos_token_id: u32, // end token id
}

impl Llama<f32> {
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
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params,
            //bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&[seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&[seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&[seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&[self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&[seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&[seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = q_buf.reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
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

            matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );

            mlp(
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
        let mut logits = Tensor::<f32>::default(&[1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &[1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &[self.d]);

        rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kvcache: &mut KVCache<f32>,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();
        let mut input = token_ids.to_vec();

        while result.len() < max_len {
            let input_tensor = Tensor::new(input.clone(), &[input.len()]);
            let res = self.forward(&input_tensor, kvcache);
            let next_token = random_sample(&res, top_p, top_k, temperature);
            if next_token == self.eos_token_id {
                break;
            }
            result.push(next_token);
            input.clear();
            input.push(next_token);
        }
        result
    }
}

#[allow(clippy::too_many_arguments)]
fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let hidden_data = unsafe { hidden_states.data_mut() };

    let score_s = 0.0f32;
    let score_d = 1.0 / (dqkv as f32).sqrt();

    //首先是QK^T和scores
    for kv_head in 0..n_kv_h {
        let scores_data = unsafe { att_scores.data_mut() };
        for q_group in 0..n_groups {
            let q_head = kv_head * n_groups + q_group;
            let q_base = q_head * dqkv;
            let k_base = kv_head * dqkv;
            for seq_pos in 0..seq_len {
                for total_pos in 0..total_seq_len {
                    let score_idx = kv_head * (n_groups * seq_len * total_seq_len)
                        + q_group * (seq_len * total_seq_len)
                        + seq_pos * total_seq_len
                        + total_pos;
                    let mut dot_product = 0.0;
                    for dim in 0..dqkv {
                        let q_idx = seq_pos * (n_kv_h * n_groups * dqkv) + q_base + dim;
                        let k_idx = total_pos * (n_kv_h * dqkv) + k_base + dim;
                        dot_product += q_data[q_idx] * k_data[k_idx];
                    }
                    scores_data[score_idx] =
                        scores_data[score_idx] * score_s + score_d * dot_product;
                }
            }
        }
    }

    masked_softmax(att_scores);

    //最后进行hidden_states处理
    let scores_data = unsafe { att_scores.data_mut() };
    for s_head in 0..n_kv_h {
        for s_group in 0..n_groups {
            let bh = s_head * n_groups + s_group;
            for seq_pos in 0..seq_len {
                for dim in 0..dqkv {
                    let hidden_idx = seq_pos * (n_kv_h * n_groups * dqkv) + bh * dqkv + dim;
                    let mut sum = 0.0;
                    for total_pos in 0..total_seq_len {
                        let score_idx =
                            bh * seq_len * total_seq_len + seq_pos * total_seq_len + total_pos;
                        let v_idx = total_pos * (n_kv_h * dqkv) + s_head * dqkv + dim;
                        sum += scores_data[score_idx] * v_data[v_idx];
                    }
                    hidden_data[hidden_idx] = sum;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    matmul_transb(up, 0.0, hidden_states, w_up, 1.0);
    swiglu(up, gate);
    let output_shape = residual.shape().clone();
    let mut output = Tensor::default(&output_shape);
    matmul_transb(&mut output, 0.0, up, w_down, 1.0);
    let size = residual.size();
    for i in 0..size {
        unsafe {
            let residual_data = residual.data_mut();
            let output_data = output.data();
            residual_data[i] += output_data[i];
        }
    }
}
