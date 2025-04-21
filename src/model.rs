use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
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
            bos_token_id: config.bos_token_id,
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
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
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

            // todo!("self_attention(...)");
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
            // todo!("down_proj matmul and add residual");
            let mut out = Tensor::default(&[seq_len, self.d]);
            OP::matmul_transb(&mut out, 0., &hidden_states, &self.params.wo[layer], 1.0); // hidden_states @ wo.T

            // 添加残差连接 (注：这里可以优化为向量加法而不是逐元素)
            let residual_data = unsafe { residual.data_mut() };
            let out_data = out.data();
            for i in 0..seq_len * self.d {
                residual_data[i] += out_data[i];
            }

            // todo!("mlp(...)");
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

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

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
        let mut result = Vec::<u32>::new();

        // todo!("实现文本生成");
        let mut cache = self.new_cache();

        // 首先，将输入token加入结果
        if !token_ids.is_empty() {
            result.extend_from_slice(token_ids);

            // 处理初始prompt
            let input_tensor = Tensor::new(token_ids.to_vec(), &[token_ids.len()]);
            let logits = self.forward(&input_tensor, &mut cache);

            // 采样第一个生成的token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(next_token);

            // 准备下一轮的输入
            let mut current_input = vec![next_token];

            // 生成剩余token直到达到最大长度或生成结束符
            while result.len() < token_ids.len() + max_len
                && *result.last().unwrap() != self.eos_token_id
            {
                let input_tensor = Tensor::new(current_input.clone(), &[1]);
                let logits = self.forward(&input_tensor, &mut cache);

                // 可以在这里添加逻辑动态调整温度以减少重复
                let adaptive_temp = if result.len() > token_ids.len() + 10 {
                    temperature * 0.9
                } else {
                    temperature
                };

                let next_token = OP::random_sample(&logits, top_p, top_k, adaptive_temp);
                result.push(next_token);
                current_input = vec![next_token];

                // 提前判断结束条件，提高性能
                if next_token == self.eos_token_id {
                    break;
                }
            }
        } else {
            // 如果没有输入token，使用BOS token作为起始
            let start_token = self.bos_token_id;
            result.push(start_token);

            let mut current_input = vec![start_token];

            // 生成token直到达到最大长度或生成结束符
            while result.len() < max_len && *result.last().unwrap() != self.eos_token_id {
                let input_tensor = Tensor::new(current_input.clone(), &[1]);
                let logits = self.forward(&input_tensor, &mut cache);

                // 动态调整temperature以减少重复
                let position = result.len();
                let adaptive_temp = if position > 10 {
                    temperature * 0.9
                } else {
                    temperature
                };

                let next_token = OP::random_sample(&logits, top_p, top_k, adaptive_temp);
                result.push(next_token);
                current_input = vec![next_token];

                // 提前判断结束条件
                if next_token == self.eos_token_id {
                    break;
                }
            }
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
    // todo!("Implement self_attention");
    // 为了数值稳定性，对注意力分数进行缩放
    let sqrt_dqkv = (dqkv as f32).sqrt();

    // 多头注意力机制的核心逻辑：
    // 1. n_kv_h: KV头的数量
    // 2. n_groups: 每个KV头对应的Q组数量，即Q头数 = n_kv_h * n_groups

    // 遍历所有KV头
    for i in 0..n_kv_h {
        let k_start = i * dqkv;
        let v_start = i * dqkv;

        // 提取当前KV头的数据
        // 将K的形状从(total_seq_len, n_kv_h * dqkv)重组为(total_seq_len, dqkv)
        let mut k_head_data = Vec::with_capacity(total_seq_len * dqkv);
        for t in 0..total_seq_len {
            let start = t * (n_kv_h * dqkv) + k_start;
            k_head_data.extend_from_slice(&k.data()[start..start + dqkv]);
        }
        let k_head = Tensor::new(k_head_data, &[total_seq_len, dqkv]);

        // 同样地，提取V头的数据
        // 将V的形状从(total_seq_len, n_kv_h * dqkv)重组为(total_seq_len, dqkv)
        let mut v_head_data = Vec::with_capacity(total_seq_len * dqkv);
        for t in 0..total_seq_len {
            let start = t * (n_kv_h * dqkv) + v_start;
            v_head_data.extend_from_slice(&v.data()[start..start + dqkv]);
        }
        let v_head = Tensor::new(v_head_data, &[total_seq_len, dqkv]);

        // 对于当前KV头，遍历所有对应的Q组
        for j in 0..n_groups {
            // 计算当前Q头的索引
            let head_idx = i * n_groups + j;
            let q_start = head_idx * dqkv;

            // 提取当前Q头的数据
            // 将Q的形状从(seq_len, n_kv_h * n_groups * dqkv)重组为(seq_len, dqkv)
            let mut q_head_data = Vec::with_capacity(seq_len * dqkv);
            for s in 0..seq_len {
                let start = s * (n_kv_h * n_groups * dqkv) + q_start;
                q_head_data.extend_from_slice(&q.data()[start..start + dqkv]);
            }
            let q_head = Tensor::new(q_head_data, &[seq_len, dqkv]);

            // 计算注意力分数: (seq_len, dqkv) @ (total_seq_len, dqkv).T -> (seq_len, total_seq_len)
            // 具体计算: scores = q_head @ k_head.T / sqrt(dqkv)
            let mut scores = Tensor::default(&[seq_len, total_seq_len]);
            OP::matmul_transb(&mut scores, 0., &q_head, &k_head, 1.0 / sqrt_dqkv);

            // 应用掩码和softmax
            // 这确保了每个token只能关注到它之前的token（因果注意力）
            OP::masked_softmax(&mut scores);

            // 计算注意力输出: (seq_len, total_seq_len) @ (total_seq_len, dqkv) -> (seq_len, dqkv)
            // 优化：使用矩阵乘法代替手动循环
            let mut attn_v = Tensor::default(&[seq_len, dqkv]);

            // 手动实现scores @ v_head矩阵乘法
            // 这里我们可以优化为使用更高效的BLAS库实现
            let scores_data = scores.data();
            let v_head_data = v_head.data();
            let attn_v_data = unsafe { attn_v.data_mut() };

            for s in 0..seq_len {
                for d in 0..dqkv {
                    let mut sum = 0.0;
                    for t in 0..total_seq_len {
                        sum += scores_data[s * total_seq_len + t] * v_head_data[t * dqkv + d];
                    }
                    attn_v_data[s * dqkv + d] = sum;
                }
            }

            // 将注意力分数存储到att_scores中，用于可能的可视化或调试
            let offset = (i * n_groups + j) * (seq_len * total_seq_len);
            unsafe {
                att_scores.data_mut()[offset..offset + seq_len * total_seq_len]
                    .copy_from_slice(scores.data());
            }

            // 将注意力输出存回hidden_states
            // hidden_states的形状是(seq_len, n_kv_h * n_groups * dqkv)
            for s in 0..seq_len {
                let start = s * (n_kv_h * n_groups * dqkv) + q_start;
                unsafe {
                    hidden_states.data_mut()[start..start + dqkv]
                        .copy_from_slice(&attn_v.data()[s * dqkv..s * dqkv + dqkv]);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mlp(
    residual: &mut Tensor<f32>,      //输入残差
    hidden_states: &mut Tensor<f32>, //隐藏层，自注意输出
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    OP::matmul_transb(gate, 0., hidden_states, w_gate, 1.);
    OP::matmul_transb(up, 0., hidden_states, w_up, 1.);
    OP::swiglu(up, gate);
    OP::matmul_transb(residual, 1., up, w_down, 1.);

    //todo!("Implement mlp");
    let total_size = residual.size();

    for i in 0..total_size {
        unsafe {
            residual.data_mut()[i] += hidden_states.data()[i];
        }
    }
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
