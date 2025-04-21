use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::operators::*;
use crate::params::LLamaParams;
use crate::tensor::{FloatConvert, Tensor};
use crate::{config::LlamaConfigJson, params::FromLeBytes};
use num_traits::Float;
use safetensors::SafeTensors;
use std::fs::File;
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign};
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
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl<
        T: Default
            + Copy
            + Sum
            + Float
            + FromLeBytes
            + Clone
            + FloatConvert
            + AddAssign
            + MulAssign
            + 'static,
    > Llama<T>
{
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
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
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
        let mut residual = Tensor::<T>::default(&[seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&[seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&[seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores = Tensor::<T>::default(&[self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&[seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&[seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);
        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = q_buf.reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(
                q,
                T::from_f32(0f32),
                &hidden_states,
                &self.params.wq[layer],
                T::from_f32(1f32),
            );
            OP::matmul_transb(
                k,
                T::from_f32(0f32),
                &hidden_states,
                &self.params.wk[layer],
                T::from_f32(1f32),
            );
            OP::matmul_transb(
                v,
                T::from_f32(0f32),
                &hidden_states,
                &self.params.wv[layer],
                T::from_f32(1f32),
            );

            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
                                                       // ======== Self-Attention ========
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

            // ======== out = attn_V @ O_weight.T ，residual = out + residual ========
            OP::matmul_transb(
                &mut residual,
                T::from_f32(1f32),
                &hidden_states,
                &self.params.wo[layer],
                T::from_f32(1f32),
            );

            // ======== MLP ========
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
        let mut logits = Tensor::<T>::default(&[1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &[1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &[self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(
            &mut logits,
            T::from_f32(0.),
            &hidden_states,
            &self.params.lm_head,
            T::from_f32(1.0),
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
        let mut result = vec![]; // 结果列表，初始化为 token_ids
        let mut kvcache = self.new_cache(); // 初始化 KVCache

        while result.len() < max_len {
            // 避免超过 max_len
            // 生成输入 token：第一步使用 token_ids，之后每次只输入上一个 token
            let input_tokens = if result.is_empty() {
                token_ids.to_vec() // 初始 prompt 作为输入
            } else {
                vec![*result.last().unwrap()] // 之后每次只输入上一个 token
            };
            let len = input_tokens.len();
            let input = Tensor::<u32>::new(input_tokens, &[len]);

            // 执行前向计算
            let logits = self.forward(&input, &mut kvcache);

            // 采样得到新 token
            let id = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(id);

            // 终止条件：如果生成了 EOS（假设 EOS = 0）
            if id == self.eos_token_id {
                break;
            }
        }
        result.pop();
        result
    }
    pub fn chat_generator(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kvcache: &mut KVCache<T>,
    ) -> Vec<u32> {
        let mut result = vec![]; // 结果列表，初始化为 token_ids

        while result.len() <= max_len {
            // 避免超过 max_len
            // 生成输入 token：第一步使用 token_ids，之后每次只输入上一个 token
            let input_tokens = if result.is_empty() {
                token_ids.to_vec() // 初始 prompt 作为输入
            } else {
                vec![*result.last().unwrap()] // 之后每次只输入上一个 token
            };
            let len = input_tokens.len();
            let input = Tensor::<u32>::new(input_tokens, &[len]);

            // 执行前向计算
            let logits = self.forward(&input, kvcache);

            // 采样得到新 token
            let id = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(id);

            // 终止条件：如果生成了 EOS（假设 EOS = 0）
            if id == self.eos_token_id {
                break;
            }
        }
        result.pop(); // 去掉 EOS
        result
        // todo!("Add new function to attach the model to the chatbot");
    }
}

#[allow(clippy::too_many_arguments)]
fn self_attention<T: Default + Copy + Float + FloatConvert + MulAssign + AddAssign + 'static>(
    hidden_states: &mut Tensor<T>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<T>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<T>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize, // n_q_h = n_kv_h * n_groups
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let scale = (dqkv as f32).sqrt();
    let _q = q.data();
    let _k = k.data();

    let q_cols = n_kv_h * n_groups * dqkv;

    // ======== Step 1 : score = Q @ K.T / sqrt(dim) 计算注意力分数 ========
    let kv_cols = n_kv_h * dqkv;
    {
        let _att_scores = unsafe { att_scores.data_mut() };
        // 遍历每个KV头
        for kv_head in 0..n_kv_h {
            // 每个KV头对应的Q头组（包含n_groups个Q头）
            let q_head_start = kv_head * n_groups;

            // 计算注意力分数
            for s in 0..seq_len {
                for g in 0..n_groups {
                    let q_head = q_head_start + g;
                    // 计算Q[s]中当前Q头的向量
                    let q_start = q_head * dqkv;
                    let q_end = q_start + dqkv;

                    let q_vec = &_q[s * q_cols + q_start..s * q_cols + q_end];

                    // 遍历K的每个位置计算点积
                    for t in 0..total_seq_len {
                        // 获取K中对应KV头的向量
                        let k_start = kv_head * dqkv;
                        let k_end = k_start + dqkv;

                        let k_vec = &_k[t * kv_cols + k_start..t * kv_cols + k_end];

                        // 手动计算点积
                        let mut dot = 0.0;
                        for i in 0..dqkv {
                            dot += <T as FloatConvert>::to_f32(q_vec[i])
                                * <T as FloatConvert>::to_f32(k_vec[i]);
                        }
                        _att_scores[kv_head * (n_groups * seq_len * total_seq_len)
                            + g * (seq_len * total_seq_len)
                            + s * total_seq_len
                            + t] = T::from_f32(dot / scale);
                    }
                }
            }
        }
    }

    // ======== Step 2: attn = masked_softmax(score) ========
    OP::masked_softmax(att_scores);

    // ======== Step 3: hidden_states = attn @ V ========
    let _att_scores = att_scores.data();
    let mut _hidden_states = unsafe { hidden_states.data_mut() };
    _hidden_states.fill(T::from_f32(0.));
    let _v = v.data();

    for kv_head in 0..n_kv_h {
        for v_id in 0..total_seq_len {
            // 优先确定V的位置，避免重复计算，减少内存访问
            let v_start = kv_head * dqkv;
            let v_end = v_start + dqkv;
            let v_vec = &_v[v_id * kv_cols + v_start..v_id * kv_cols + v_end];

            for g in 0..n_groups {
                let q_head = kv_head * n_groups + g;
                let q_start = q_head * dqkv;

                for s in 0..seq_len {
                    let score = _att_scores[kv_head * (n_groups * seq_len * total_seq_len)
                        + g * (seq_len * total_seq_len)
                        + s * total_seq_len
                        + v_id];
                    let out_start = s * q_cols + q_start;
                    for i in 0..dqkv {
                        _hidden_states[out_start + i] += score * v_vec[i];
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mlp<T: Default + Copy + Float + FloatConvert + AddAssign + MulAssign + 'static>(
    residual: &mut Tensor<T>,
    hidden_states: &mut Tensor<T>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: f32,
) {
    //RMS 归一化
    rms_norm(hidden_states, residual, rms_w, eps);

    // Compute gate = hidden_states @ w_gate.T
    matmul_transb(
        gate,
        T::from_f32(0.),
        hidden_states,
        w_gate,
        T::from_f32(1.0),
    );

    // Step 3: Compute up = hidden_states @ w_up.T
    matmul_transb(up, T::from_f32(0.), hidden_states, w_up, T::from_f32(1.0));

    // Step 4: Compute act = gate * sigmoid(gate) * up using SwiGLU
    // Calculate sigmoid of gate into a temporary tensor
    // 这里是SiLU（gate）
    swiglu(up, gate);
    //Step 5: residual = hidden_states + residual
    matmul_transb(residual, T::from_f32(1.0), up, w_down, T::from_f32(1.0));
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
    mlp(
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
