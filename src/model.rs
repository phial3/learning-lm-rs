use std::cmp::{min, Ordering};
use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::operators::*;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize, // number of heads for q  // Q头数量是KV头的整数倍，所以nqh = nkvh * n_groups
    n_kv_h: usize, // number of heads for k and v
    d: usize,     // dimension of hidden states
    dqkv: usize,  // length of a single q, k, or v vector
    di: usize,    // dimension of intermediate states
    eps: f32,     // epsilon for RMS normalization
    rope_theta: f32, // rope theta for rope initialization
    max_seq_len: usize, // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32, // start token id
    eos_token_id: u32, // end token id
}

fn easy_softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = *logits
        .iter()
        .reduce(|a, b| if a > b { a } else { b })
        .unwrap();
    let exp_logits: Vec<_> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp_logits: f32 = exp_logits.iter().sum();
    let result: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp_logits).collect();
    result.to_vec()
}

macro_rules! flush_print {
    ($fmt:expr) => ({
        use std::io;
        use std::io::Write;
        let mut stdout = io::stdout();
        write!(stdout, $fmt).unwrap();
        stdout.flush().unwrap();
    });
    ($fmt:expr, $($arg:tt)*) => ({
        use std::io;
        use std::io::Write;
        let mut stdout = io::stdout();
        write!(stdout, $fmt, $($arg)*).unwrap();
        stdout.flush().unwrap();
    });
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

    #[allow(dead_code)]
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
            // out = attn_V @ O_weight( self.params.wo[layer] ).T
            // residual = out + residual
            matmul_transb(
                &mut residual,
                1f32,
                &hidden_states,
                &self.params.wo[layer],
                1f32,
            );

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

        logits // (1, vocab_size)
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        // let mut result = Vec::<u32>::new();
        let mut result = Vec::<u32>::from(token_ids);

        result.push(self.bos_token_id);

        let mut input = result.clone();

        // todo!("实现文本生成");
        let mut cache = KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0);

        while result.len() < max_len {
            // 第一次输入全部
            let mut logits = self.forward(
                &Tensor::<u32>::new(input.clone(), &[input.len()]),
                &mut cache,
            );
            let length = logits.size();
            let data = unsafe { logits.data_mut() };
            if temperature > 0. {
                for item in data.iter_mut().take(length) {
                    *item /= temperature;
                }
            }
            let logits = easy_softmax(logits.data());
            let new_word_id = Self::select_word_to_id(&logits.to_vec(), top_p, top_k as usize);
            // let new_word_id = random_sample(&logits, top_p, top_k, temperature);  // 写完才发现原来OP里这已经写好了……
            if new_word_id == self.eos_token_id {
                break;
            }
            result.push(new_word_id);
            input = vec![new_word_id];

            // // 调试-B
            use std::path::PathBuf;
            use tokenizers::Tokenizer;
            let project_dir = env!("CARGO_MANIFEST_DIR");
            let model_dir = PathBuf::from(project_dir).join("models").join("story");
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            // println!(
            //     "new word id: [{}]，curr length: [{}]，new word is [{}]",
            //     new_word_id, result.len(), tokenizer.decode(&[new_word_id], true).unwrap()
            // );
            // 逐步输出-B
            flush_print!("{}", tokenizer.decode(&[new_word_id], true).unwrap());
            // 逐步输出-E
            // // 调试-E
        }

        if result.len() == max_len {
            println!(" <|heke: is max len. its maybe has some error|>");
        }

        result
    }

    fn select_word_to_id(logits: &[f32], top_p: f32, top_k: usize) -> u32 {
        let mut indices_and_values: Vec<(_, _)> = logits
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, value))
            .collect();

        indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut take_num = 1; // take_num 最小是1
        let mut tmp_sump = 0.;

        if top_p > 0. {
            for i in indices_and_values.iter() {
                tmp_sump += i.1;
                if tmp_sump > top_p {
                    break;
                }
                take_num += 1;
            }
        }

        take_num = if top_k > 0 {
            min(take_num, top_k)
        } else {
            take_num
        };

        let top_indices: Vec<f32> = indices_and_values
            .iter()
            .cloned()
            .take(take_num)
            .map(|(_, w)| w)
            .collect();

        let resampled = easy_softmax(&top_indices);

        use rand::distributions::Distribution;
        use rand::distributions::WeightedIndex;

        let mut rng = rand::thread_rng();
        let index = WeightedIndex::new(resampled).unwrap().sample(&mut rng);

        indices_and_values[index].0 as u32
    }
}

#[allow(clippy::too_many_arguments)]
fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,                   // number of heads for k and v
    n_groups: usize,                 // n_head_of_q / n_head_of_k
    seq_len: usize,                  // input len of seq-token-vec
    total_seq_len: usize,            // len of kv-cache + seq_len
    dqkv: usize,                     // length of a single q, k, or v vector，人话：dim
) {
    // todo!("Implement self_attention");
    // ### 以下是你需要实现的部分
    // score = Q @ K.T / sqrt(dim)
    // attn = softmax(score)
    // attn_V = attn @ V
    let bias: usize = seq_len * total_seq_len;
    let (beta, alpha) = (0f32, 1.0 / ((dqkv as f32).sqrt()));
    let (_m, _k, _n) = (seq_len, dqkv, total_seq_len);
    for head in 0..n_kv_h {
        for group in 0..n_groups {
            // 这里的bh是针对scores的，对于QK不适用
            let bh = n_groups * head + group;
            // 此时取保证”假设“取出的Q和K是(seq_len, dim) 和 (total_seq_len, dim)
            let (_a, _b, _c) = (q.data(), k.data(), unsafe { att_scores.data_mut() });
            let clip_a = |i, j| {
                _a[i * n_kv_h * n_groups * _k + bh * _k + j] // 索引Q的节点
            };
            let clip_b = |i, j| {
                _b[i * n_kv_h * _k + head * _k + j] // 索引K的节点
            };
            // 完成索引后就是简单的求积之和
            for mi in 0.._m {
                for ni in 0.._n {
                    let idx = bh * bias + mi * total_seq_len + ni;
                    _c[idx] *= beta; // 乘0置零
                    _c[idx] += alpha
                        * (0.._k)
                            .map(|ki| clip_a(mi, ki) * clip_b(ni, ki))
                            .sum::<f32>();
                }
            }
        }
    }

    masked_softmax(att_scores);

    let (_a, _b, _c) = (att_scores.data(), v.data(), unsafe {
        hidden_states.data_mut()
    });
    for head in 0..n_kv_h {
        for group in 0..n_groups {
            // 这里开始算 attn_V = attn @ V =》 hidden = scores @ v
            let bh = n_groups * head + group;
            // 这里毕竟熟练了，就不写匿名函数索引了，直接写吧……
            for i in 0..seq_len {
                for j in 0..dqkv {
                    let idx = i * n_kv_h * n_groups * dqkv + bh * dqkv + j;
                    _c[idx] = (0..total_seq_len)
                        .map(|idx| {
                            _a[bh * bias + i * total_seq_len + idx]
                                * _b[head * dqkv + j + idx * n_kv_h * dqkv]
                        })
                        .sum::<f32>();
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mlp(
    residual: &mut Tensor<f32>,      // x 以及之后的 y
    hidden_states: &mut Tensor<f32>, // 用于存储过程中的计算结果
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // todo!("Implement mlp");
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0f32, hidden_states, w_gate, 1f32);
    matmul_transb(up, 0f32, hidden_states, w_up, 1f32);
    swiglu(up, gate);
    matmul_transb(residual, 1f32, up, w_down, 1f32);
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
