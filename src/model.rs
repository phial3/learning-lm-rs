use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::operators::{matmul_transb, rms_norm, swiglu};
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
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0); //Q = RoPE(x @ Q_weight.T)
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0); //K = RoPE(x @ K_weight.T)
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0); //V = x @ V_weight.T
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            ); //Q = RoPE(x @ Q_weight.T)
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            ); //K = RoPE(x @ K_weight.T)

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)  K = cat(K_cache, K)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)  V = cat(V_cache, V)

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
            OP::matmul_transb(
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
            // todo!("mlp(...)");
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
        let mut result = vec![self.bos_token_id]; // 初始化 bos 作为起点
        let mut cache = self.new_cache(); // 初始化 KVCache

        let mut token_iter = token_ids.iter().cloned(); // 创建 token 迭代器
        let mut current_token = self.bos_token_id; // 记录当前 token

        // **统一的 while 循环**
        while result.len() < max_len {
            // **如果还有用户输入的 token，优先使用它，否则用上一个生成的 token**
            let input_token = token_iter.next().unwrap_or(current_token);

            let input_tensor = Tensor::new(vec![input_token], &[1]); // 只传入一个 token
            let logits = self.forward(&input_tensor, &mut cache); // 计算 logits

            // **如果仍在处理用户输入的 token，直接使用它，否则进行采样**
            current_token = if token_iter.len() > 0 {
                input_token
            } else {
                OP::random_sample(&logits, top_p, top_k, temperature)
            };

            if current_token == self.eos_token_id {
                break;
            }

            result.push(current_token);
        }
        // todo!("实现文本生成");
        result
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn self_attention(
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
    {
        let att_data = unsafe { att_scores.data_mut() };
        for kv_head in 0..n_kv_h {
            for group in 0..n_groups {
                for seq in 0..seq_len {
                    for total_seq in 0..total_seq_len {
                        let mut score = 0.0;
                        for dim in 0..dqkv {
                            let q_index = seq * (n_kv_h * n_groups * dqkv)
                                + (kv_head * n_groups + group) * dqkv
                                + dim;
                            let k_index = total_seq * (n_kv_h * dqkv) + kv_head * dqkv + dim;
                            score += q.data()[q_index] * k.data()[k_index];
                        }
                        score /= (dqkv as f32).sqrt();
                        att_data[(kv_head * n_groups + group) * seq_len * total_seq_len
                            + seq * total_seq_len
                            + total_seq] = score;
                    }
                }
            }
        }
    }
    OP::masked_softmax(att_scores);
    // 计算 attn_V = attn @ V
    //  let mut attn_v = Tensor::<f32>::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);
    //  let attn_v_data = unsafe { attn_v.data_mut() };
    let hidden_data = unsafe { hidden_states.data_mut() };
    let att_data = unsafe { att_scores.data_mut() };
    for i in 0..hidden_data.len() {
        hidden_data[i] = 0.0;
    }
    for kv_head in 0..n_kv_h {
        for group in 0..n_groups {
            for seq in 0..seq_len {
                for dim in 0..dqkv {
                    let mut attn_v_sum = 0.0;
                    for total_seq in 0..total_seq_len {
                        let att_score_index =
                            (kv_head * n_groups + group) * seq_len * total_seq_len
                                + seq * total_seq_len
                                + total_seq;
                        let v_index = total_seq * (n_kv_h * dqkv) + kv_head * dqkv + dim;
                        attn_v_sum += att_data[att_score_index] * v.data()[v_index];
                    }
                    let attn_v_index = seq * (n_kv_h * n_groups * dqkv)
                        + (kv_head * n_groups + group) * dqkv
                        + dim;
                    hidden_data[attn_v_index] = attn_v_sum;
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
    // todo!("Implement mlp");
    let new_shape = residual.shape().clone();
    let num_elements = new_shape.iter().product::<usize>();
    let new_data = vec![0.0; num_elements];
    let mut output = Tensor::<f32>::new(new_data, &new_shape);

    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0., hidden_states, w_gate, 1.);
    matmul_transb(up, 0., hidden_states, w_up, 1.);
    swiglu(up, gate);
    matmul_transb(&mut output, 0., up, w_down, 1.);

    let len = residual.size();
    let _residual = unsafe { residual.data_mut() };
    let _output = output.data();
    for i in 0..len {
        _residual[i] += _output[i];
    }
    // println!("{:?}", residual.data());
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
