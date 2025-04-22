use std::fs::File;
use std::sync::Arc;
use std::time::Instant;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, masked_softmax, matmul_transb, random_sample, rms_norm, swiglu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use std::path::Path;
use std::thread;
use num_cpus;
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
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        #[cfg(not(feature = "distributed"))]
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

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

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
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

            #[cfg(feature = "distributed")]
            self_attention_distributed(&mut residual, &mut hidden_states, q, full_k, full_v, &self.params.wo[layer],
                self.n_kv_h, n_groups, seq_len, total_seq_len, self.dqkv, self.d);
            #[cfg(not(feature = "distributed"))]
            self_attention(&mut hidden_states, &mut att_scores, q, full_k, full_v,
                self.n_kv_h, n_groups, seq_len, total_seq_len, self.dqkv);
            
            #[cfg(not(feature = "distributed"))]
            OP::matmul_transb(&mut residual, 1., &hidden_states, &self.params.wo[layer], 1.);
            
            #[cfg(feature = "distributed")]
            mlp_distributed(&mut residual, &mut hidden_states, &mut gate_buf, &mut up_buf, &self.params.w_up[layer], &self.params.w_down[layer],
                 &self.params.w_gate[layer], &self.params.rms_ffn_w[layer], self.eps, self.di, seq_len, self.d);
            #[cfg(not(feature = "distributed"))]
            mlp(&mut residual, &mut hidden_states, &mut gate_buf, &mut up_buf, &self.params.w_up[layer], &self.params.w_down[layer],
                &self.params.w_gate[layer], &self.params.rms_ffn_w[layer], self.eps);
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

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
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        // todo!("实现文本生成");
        let mut kvcache = self.new_cache();
        let mut current_token_ids = token_ids.to_vec();
        for _ in 0..max_len {
            let shape = current_token_ids.len();
            let output_ids = self.forward(&Tensor::new(current_token_ids, &vec![shape]), &mut kvcache);
            let output = random_sample(&output_ids, top_p, top_k, temperature);
            if output == self.eos_token_id { break; }
            result.push(output);
            current_token_ids = vec![output];
        }
        result
    }
    
    pub fn chat_generate(
        &self,
        tokenizer: Tokenizer,
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) {
        let mut kvcache = self.new_cache();
        loop {
            println!("User:");
            let mut result = Vec::<u32>::new();
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            input.pop();
            if input == "\\q".to_string() {
                return;
            }
            let start = Instant::now();
            let render_input = String::from("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n")
            + &input + "<|im_end|>\n<|im_start|>assistant";
            // println!("Input: {render_input}");
            let binding = tokenizer.encode(render_input, true).unwrap();
            let input_ids = binding.get_ids();
            let mut current_token_ids = input_ids.to_vec();
            // one time generate
            for _ in 0..max_len {
                let shape = current_token_ids.len();
                let output_ids = self.forward(&Tensor::new(current_token_ids, &vec![shape]), &mut kvcache);
                let output = random_sample(&output_ids, top_p, top_k, temperature);
                if output == self.eos_token_id { break; }
                result.push(output);
                current_token_ids = vec![output];
            }
            println!("AI: {}", tokenizer.decode(&result, true).unwrap());
            let duration = start.elapsed();
            println!("Time elapsed is: {:?}", duration);
        }
    }
}

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
    let _q = q.data();
    let _k = k.data();
    let _v = v.data();
    let _hidden_states = unsafe { hidden_states.data_mut() };
    for i in 0..n_kv_h {
        for j in 0..n_groups {
            let mut score  =att_scores.slice((i * n_groups + j) * seq_len * total_seq_len, 
                &vec![seq_len, total_seq_len]);
            let _score = unsafe { score.data_mut() };
            // 向量乘法
            for k in 0..seq_len {
                for l in 0..total_seq_len {
                    let mut sum = 0.;
                    for m in 0..dqkv {
                        sum += _q[k * n_kv_h * n_groups * dqkv + (i * n_groups + j) * dqkv + m] 
                            * _k[l * n_kv_h * dqkv + i * dqkv + m];
                    }
                    _score[k * total_seq_len + l] = sum;
                }
            }
            // 除以sqrt(dim)
            for i in 0..seq_len {
                for j in 0..total_seq_len {
                    _score[i * total_seq_len + j] /= (dqkv as f32).sqrt();
                }
            }
            masked_softmax(&mut score);
            let _score = unsafe { score.data_mut() };
            // attn_V(seq * dqkv) = attn(seq * total_seq) @ V(total_seq * dqkv)
            for k in 0..seq_len {
                for l in 0..dqkv {
                    let mut sum = 0.;
                    for m in 0..total_seq_len {
                         sum += _score[k * total_seq_len + m] 
                            * _v[m * n_kv_h * dqkv + i * dqkv + l];
                    }
                    _hidden_states[k * n_kv_h * n_groups * dqkv + (i * n_groups + j) * dqkv + l] = sum;
                }
            }
        }
    }
}

fn self_attention_distributed(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    // att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    wo: &Tensor<f32>,
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
    d: usize,
) {
    if n_kv_h * n_groups != num_cpus::get() {
        eprintln!("Warning: Here we assume that the number of CPU cores is equal to the number of heads, but currently they are not equal.");
    }
    let _q = Arc::new(q.clone());
    let _k = Arc::new(k.clone());
    let _v = Arc::new(v.clone());
    let _wo = Arc::new(wo.clone());
    let thread_count = n_kv_h * n_groups;
    let handles: Vec<_> = (1..thread_count).map(|i| {
        let __q = _q.clone();
        let __k = _k.clone();
        let __v = _v.clone();
        let __wo = _wo.clone();

        thread::spawn(move || {
            let ___q = __q.data();
            let ___k = __k.data();
            let ___v = __v.data();
            let ___wo = __wo.data();

            let mut score: Tensor<f32> = Tensor::default(&vec![seq_len, total_seq_len]);
            let _score = unsafe { score.data_mut() };
            let mut residual = Tensor::<f32>::default(&vec![seq_len, d]);
            let _residual = unsafe { residual.data_mut() };
            let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
            let _hidden_states = unsafe { hidden_states.data_mut() };
            let kv_index = i / n_groups;
            let n_group = i % n_groups;

            for k in 0..seq_len {
                for l in 0..total_seq_len {
                    let mut sum = 0.;
                    for m in 0..dqkv {
                        sum += ___q[k * n_kv_h * n_groups * dqkv + (kv_index * n_groups + n_group) * dqkv + m] 
                            * ___k[l * n_kv_h * dqkv + kv_index * dqkv + m];
                    }
                    _score[k * total_seq_len + l] = sum;
                }
            }
            // 除以sqrt(dim)
            for m in 0..seq_len {
                for n in 0..total_seq_len {
                    _score[m * total_seq_len + n] /= (dqkv as f32).sqrt();
                }
            }
            masked_softmax(&mut score);

            let _score = unsafe { score.data_mut() };
            // attn_V(seq * dqkv) = attn(seq * total_seq) @ V(total_seq * dqkv)
            for k in 0..seq_len {
                for l in 0..dqkv {
                    let mut sum = 0.;
                    for m in 0..total_seq_len {
                         sum += _score[k * total_seq_len + m] 
                            * ___v[m * n_kv_h * dqkv + kv_index * dqkv + l];
                    }
                    _hidden_states[k * n_kv_h * n_groups * dqkv + i * dqkv + l] = sum;
                }
            }

            // residual(seq * hidden_size) = attn_V(seq * dqkv) @ wo(dqkv * hidden_size)
            for m in 0..seq_len {
                for n in 0..d {
                    let mut sum = 0.0;
                    for o in 0..dqkv {
                        sum += _hidden_states[m * n_kv_h * n_groups * dqkv + i * dqkv + o] 
                            * ___wo[n * n_kv_h * n_groups * dqkv + i * dqkv + o];
                    }
                    _residual[m * d + n] = sum;
                }
            }
            residual
        })
    }).collect();

    let __q = _q.data();
    let __k = _k.data();
    let __v = _v.data();
    let __wo = _wo.data();

    let mut score: Tensor<f32> = Tensor::default(&vec![seq_len, total_seq_len]);
    let _score = unsafe { score.data_mut() };

    for k in 0..seq_len {
        for l in 0..total_seq_len {
            let mut sum = 0.;
            for m in 0..dqkv {
                sum += __q[k * n_kv_h * n_groups * dqkv + m] 
                    * __k[l * n_kv_h * dqkv + m];
            }
            _score[k * total_seq_len + l] = sum;
        }
    }
    // 除以sqrt(dim)
    for m in 0..seq_len {
        for n in 0..total_seq_len {
            _score[m * total_seq_len + n] /= (dqkv as f32).sqrt();
        }
    }
    masked_softmax(&mut score);

    let _score = unsafe { score.data_mut() };
    let _hidden_states = unsafe { hidden_states.data_mut() };
    // attn_V(seq * dqkv) = attn(seq * total_seq) @ V(total_seq * dqkv)
    for k in 0..seq_len {
        for l in 0..dqkv {
            let mut sum = 0.;
            for m in 0..total_seq_len {
                 sum += _score[k * total_seq_len + m] 
                    * __v[m * n_kv_h * dqkv + l];
            }
            _hidden_states[k * n_kv_h * n_groups * dqkv + l] = sum;
        }
    }

    // residual(seq * hidden_size) = attn_V(seq * dqkv) @ wo(dqkv * hidden_size)
    let _residual = unsafe { residual.data_mut() };
    for m in 0..seq_len {
        for n in 0..d {
            let mut sum = 0.0;
            for o in 0..dqkv {
                sum += _hidden_states[m * n_kv_h * n_groups * dqkv + o] 
                    * __wo[n * n_kv_h * n_groups * dqkv + o];
            }
            _residual[m * d + n] += sum;
        }
    }
    for (_, handle) in handles.into_iter().enumerate() {
        let distributed_residual = handle.join().unwrap();
        residual.all_reduce(distributed_residual);
    }
}

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
    // hidden = rms_norm(residual)
    rms_norm(hidden_states, residual, rms_w, eps);

    // gate = hidden @ gate_weight.T
    matmul_transb(gate, 0., hidden_states, w_gate, 1.);

    // up = hidden @ up_weight.T
    matmul_transb(up, 0., hidden_states, w_up, 1.);

    // act = gate * sigmoid(gate) * up ## SwiGLU
    swiglu(up, gate);

    // output = act @ down_weight.T
    // residual = output + residual
    matmul_transb(residual, 1., up, w_down, 1.);
    
    // todo!("Implement mlp");
}

fn mlp_distributed(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
    di: usize,
    seq_len: usize,
    d: usize,
) {
    // hidden = rms_norm(residual)
    rms_norm(hidden_states, residual, rms_w, eps);

    let thread_count = num_cpus::get();
    assert!(di % thread_count == 0, "expected cpu corrs:{}, but got di:{}", thread_count, di);

    let _hidden_states = Arc::new(hidden_states.clone());
    let _w_gate = Arc::new(w_gate.clone());
    let _w_up = Arc::new(w_up.clone());
    let _w_down = Arc::new(w_down.clone());

    let handles: Vec<_> = (1..thread_count).map(|i| {
        let __hidden_states = _hidden_states.clone();
        let __w_gate = _w_gate.slice(di / thread_count * d * i, &vec![di / thread_count, d]);
        let __w_up = _w_up.slice(di / thread_count * d * i, &vec![di / thread_count, d]);
        let __w_down = _w_down.clone();
        thread::spawn(move|| {
            let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di / thread_count]);
            let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di / thread_count]);
            // gate = hidden @ gate_weight.T
            matmul_transb(&mut gate_buf, 0., &__hidden_states, &__w_gate, 1.);

            // up = hidden @ up_weight.T
            matmul_transb(&mut up_buf, 0., &__hidden_states, &__w_up, 1.);

            // act = gate * sigmoid(gate) * up ## SwiGLU
            swiglu(&mut up_buf, &gate_buf);

            // output = act @ down_weight.T
            let mut residual: Tensor<f32> = Tensor::default(&vec![seq_len, d]);
            let _residual = unsafe { residual.data_mut() };
            let _up_buf = up_buf.data();
            let ___w_down = __w_down.data();
            for m in 0..seq_len {
                for n in 0..d {
                    let mut sum = 0.0 as f32;
                    for o in 0..di / thread_count {
                        sum += _up_buf[m * di / thread_count + o] 
                            * ___w_down[n * di + i * di / thread_count + o];
                    }
                    _residual[m * d + n] = sum;
                }
            }
            residual
        })
    }).collect();

    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di / thread_count]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di / thread_count]);

    let __w_gate = _w_gate.slice(0, &vec![di / thread_count, d]);
    let __w_up = _w_up.slice(0, &vec![di / thread_count, d]);

    matmul_transb(&mut gate_buf, 0., hidden_states, &__w_gate, 1.);

    // up = hidden @ up_weight.T
    matmul_transb(&mut up_buf, 0., hidden_states, &__w_up, 1.);

    // act = gate * sigmoid(gate) * up ## SwiGLU
    swiglu(&mut up_buf, &gate_buf);

    // output = act @ down_weight.T
    let _residual = unsafe { residual.data_mut() };
    let _up_buf = up_buf.data();
    let ___w_down = w_down.data();
    for m in 0..seq_len {
        for n in 0..d {
            let mut sum = 0.0 as f32;
            for o in 0..di / thread_count {
                sum += _up_buf[m * di / thread_count + o] 
                    * ___w_down[n * di + o];
            }
            _residual[m * d + n] += sum;
        }
    }
    // residual = output + residual
    for (_, handle) in handles.into_iter().enumerate() {
        let distributed_residual = handle.join().unwrap();
        residual.all_reduce(distributed_residual);
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
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    
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

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
