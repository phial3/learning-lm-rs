use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::ChatParams;
use crate::tensor::Tensor;
use std::io::{self, Write};
use tokenizers::Tokenizer;

pub struct ChatHistory {
    messages: Vec<Message>,
    system_message: String,
}

#[derive(Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl ChatHistory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            system_message: "You are a helpful AI assistant. Please provide clear, concise, and accurate responses.".to_string(),
        }
    }

    pub fn set_system_message(&mut self, message: String) {
        self.system_message = message;
    }

    pub fn add_message(&mut self, role: String, content: String) {
        self.messages.push(Message { role, content });
    }

    pub fn format_prompt(&self) -> String {
        let mut prompt = String::new();

        // 添加系统消息
        prompt.push_str(&format!(
            "<|im_start|>system\n{}\n<|im_end|>\n",
            self.system_message
        ));

        // 添加历史消息
        for message in &self.messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}\n<|im_end|>\n",
                message.role, message.content
            ));
        }

        // 添加助手回复开始标记
        prompt.push_str("<|im_start|>assistant\n");

        prompt
    }

    #[allow(dead_code)]
    pub fn clear_history(&mut self) {
        self.messages.clear();
    }
}

pub struct Chat {
    params: ChatParams<f32>,
    pub tokenizer: Tokenizer,
    config: LlamaConfigJson,
    history: ChatHistory,
    kv_cache: KVCache<f32>,
    n_layers: usize,
    max_seq_len: usize,
    n_kv_h: usize,
    d: usize,
    eps: f32,
    rope_theta: f32,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl Chat {
    pub fn new(params: ChatParams<f32>, tokenizer: Tokenizer, config: LlamaConfigJson) -> Self {
        let n_layers = config.num_hidden_layers;
        let max_seq_len = config.max_position_embeddings;
        let n_kv_h = config.num_key_value_heads;
        let d = config.hidden_size;
        let eps = config.rms_norm_eps;
        let rope_theta = config.rope_theta;
        let bos_token_id = config.bos_token_id;
        let eos_token_id = config.eos_token_id;

        let kv_cache = KVCache::new(
            n_layers,
            max_seq_len,
            n_kv_h * (d / config.num_attention_heads),
            0,
        );

        Self {
            params,
            tokenizer,
            config,
            history: ChatHistory::new(),
            kv_cache,
            n_layers,
            max_seq_len,
            n_kv_h,
            d,
            eps,
            rope_theta,
            bos_token_id,
            eos_token_id,
        }
    }

    pub fn set_system_message(&mut self, message: String) {
        self.history.set_system_message(message);
    }

    pub fn chat(&mut self, user_input: &str) -> String {
        // 添加用户输入到历史记录
        self.history
            .add_message("user".to_string(), user_input.to_string());

        // 生成回复
        let prompt = self.history.format_prompt();
        let tokens = self.tokenizer.encode(prompt.as_str(), true).unwrap();
        let input_ids = tokens.get_ids();

        let mut result = Vec::<u32>::new();
        result.extend_from_slice(input_ids);

        let input_tensor = Tensor::new(input_ids.to_vec(), &[input_ids.len()]);
        let logits = self.forward(&input_tensor);

        let next_token = OP::random_sample(&logits, 0.9, 50, 0.7);
        result.push(next_token);

        let mut token_count = 1;
        let mut generated_tokens = Vec::<u32>::new();
        generated_tokens.push(next_token);

        // 获取一个完整句子需要的token数量
        let sentence_tokens = 10;
        let mut last_output = String::new();

        let mut current_input = vec![next_token];
        let mut no_progress_count = 0;
        let max_no_progress = 5;
        let mut last_token = next_token;
        let mut repetition_penalty = 1.0;

        while result.len() < input_ids.len() + 1000 && *result.last().unwrap() != self.eos_token_id
        {
            let input_tensor = Tensor::new(current_input.clone(), &[1]);
            let logits = self.forward(&input_tensor);

            let adaptive_temp = if result.len() > input_ids.len() + 20 {
                0.7 * 0.8
            } else {
                0.7
            };

            let next_token =
                OP::random_sample(&logits, 0.9, 50, adaptive_temp * repetition_penalty);
            token_count += 1;
            // println!("Token ID: {}, 序列长度: {}", next_token, result.len() + 1);

            generated_tokens.push(next_token);

            if generated_tokens.len() >= sentence_tokens || next_token == self.eos_token_id {
                if let Ok(new_text) = self.tokenizer.decode(&generated_tokens, true) {
                    if !new_text.contains("<|im_end|>") {
                        // 获取新增部分
                        if !last_output.is_empty() {
                            // 防止重复输出，只显示新部分
                            if new_text.len() > last_output.len() {
                                let new_part = &new_text[last_output.len()..];
                                print!("{}", new_part);
                                io::stdout().flush().unwrap();
                            }
                        } else {
                            print!("{}", new_text);
                            io::stdout().flush().unwrap();
                        }
                        last_output = new_text;
                    }
                }
            }

            if next_token == last_token {
                no_progress_count += 1;
                repetition_penalty *= 1.2;
                if no_progress_count >= max_no_progress {
                    break;
                }
            } else {
                no_progress_count = 0;
                repetition_penalty = 1.0;
                last_token = next_token;
            }

            result.push(next_token);
            current_input = vec![next_token];

            if next_token == self.eos_token_id {
                break;
            }
        }

        let response = self
            .tokenizer
            .decode(&result[input_ids.len()..], true)
            .unwrap()
            .replace("<|im_end|>", "")
            .trim()
            .to_string();

        println!("\n\n总生成 token 数: {}", token_count);
        println!("总序列长度: {}", result.len());

        self.history
            .add_message("assistant".to_string(), response.clone());

        response
    }

    fn forward(&mut self, input: &Tensor<u32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = self.kv_cache.len();
        self.kv_cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.config.num_attention_heads / self.n_kv_h;

        // 预分配缓冲区
        let mut residual = Tensor::<f32>::default(&[seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&[seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&[
            seq_len,
            self.config.num_attention_heads * (self.d / self.config.num_attention_heads),
        ]);
        let mut att_scores =
            Tensor::<f32>::default(&[self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&[seq_len, self.config.intermediate_size]);
        let mut up_buf = Tensor::<f32>::default(&[seq_len, self.config.intermediate_size]);

        // 计算开始
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = q_buf.reshape(&vec![
                seq_len,
                self.config.num_attention_heads * (self.d / self.config.num_attention_heads),
            ]);
            let k = &mut self.kv_cache.k_cache(layer, past_seq_len);
            let v = &mut self.kv_cache.v_cache(layer, past_seq_len);

            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);

            OP::rope(
                q.reshape(&vec![
                    seq_len,
                    self.config.num_attention_heads,
                    self.d / self.config.num_attention_heads,
                ]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![
                    seq_len,
                    self.n_kv_h,
                    self.d / self.config.num_attention_heads,
                ]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut self.kv_cache.k_cache(layer, 0);
            let full_v = &mut self.kv_cache.v_cache(layer, 0);

            self.self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                seq_len,
                total_seq_len,
            );

            let mut out = Tensor::default(&[seq_len, self.d]);
            OP::matmul_transb(&mut out, 0., &hidden_states, &self.params.wo[layer], 1.0);

            unsafe {
                let residual_data = residual.data_mut();
                let out_data = out.data();
                for i in 0..seq_len * self.d {
                    residual_data[i] += out_data[i];
                }
            }

            self.mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                layer,
            );
        }

        let mut logits = Tensor::<f32>::default(&[1, self.config.vocab_size]);
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

    pub fn generate(&mut self, token_ids: &[u32]) -> String {
        let mut result = Vec::<u32>::new();
        let max_len = 500; // 增加最大长度
        let top_p = 0.9; // 提高 top_p
        let top_k = 50; // 增加 top_k
        let temperature = 0.7; // 降低温度，使输出更稳定

        if !token_ids.is_empty() {
            result.extend_from_slice(token_ids);

            let input_tensor = Tensor::new(token_ids.to_vec(), &[token_ids.len()]);
            let logits = self.forward(&input_tensor);

            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            println!("Generated first token: {}", next_token);
            result.push(next_token);

            let mut current_input = vec![next_token];
            let mut no_progress_count = 0;
            let max_no_progress = 5;
            let mut last_token = next_token;
            let mut repetition_penalty = 1.0;

            while result.len() < token_ids.len() + max_len
                && *result.last().unwrap() != self.eos_token_id
            {
                let input_tensor = Tensor::new(current_input.clone(), &[1]);
                let logits = self.forward(&input_tensor);

                // 动态调整温度
                let position = result.len();
                let adaptive_temp = if position > 20 {
                    temperature * 0.8 // 降低温度
                } else {
                    temperature
                };

                // 应用重复惩罚
                let next_token =
                    OP::random_sample(&logits, top_p, top_k, adaptive_temp * repetition_penalty);

                if next_token == last_token {
                    no_progress_count += 1;
                    repetition_penalty *= 1.2; // 增加重复惩罚
                    println!("Warning: Generated same token {} times", no_progress_count);
                    if no_progress_count >= max_no_progress {
                        println!("Generation might be stuck, stopping");
                        break;
                    }
                } else {
                    no_progress_count = 0;
                    repetition_penalty = 1.0; // 重置重复惩罚
                    last_token = next_token;
                }

                result.push(next_token);
                current_input = vec![next_token];

                if next_token == self.eos_token_id {
                    println!("Generated end token, stopping");
                    break;
                }
            }
        } else {
            println!("No input tokens, using start token");
            let start_token = self.bos_token_id;
            result.push(start_token);

            let mut current_input = vec![start_token];
            let mut no_progress_count = 0;
            let max_no_progress = 5;
            let mut last_token = start_token;
            let mut repetition_penalty = 1.0;

            while result.len() < max_len && *result.last().unwrap() != self.eos_token_id {
                let input_tensor = Tensor::new(current_input.clone(), &[1]);
                let logits = self.forward(&input_tensor);

                let position = result.len();
                let adaptive_temp = if position > 20 {
                    temperature * 0.8
                } else {
                    temperature
                };

                let next_token =
                    OP::random_sample(&logits, top_p, top_k, adaptive_temp * repetition_penalty);
                println!(
                    "Generated token: {}, current length: {}, cache length: {}",
                    next_token,
                    result.len(),
                    self.kv_cache.len()
                );

                if next_token == last_token {
                    no_progress_count += 1;
                    repetition_penalty *= 1.2;
                    println!("Warning: Generated same token {} times", no_progress_count);
                    if no_progress_count >= max_no_progress {
                        println!("Generation might be stuck, stopping");
                        break;
                    }
                } else {
                    no_progress_count = 0;
                    repetition_penalty = 1.0;
                    last_token = next_token;
                }

                result.push(next_token);
                current_input = vec![next_token];

                if next_token == self.eos_token_id {
                    println!("Generated end token, stopping");
                    break;
                }
            }
        }

        println!("Generation complete, total token count: {}", result.len());
        self.tokenizer.decode(&result, true).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn self_attention(
        &self,
        hidden_states: &mut Tensor<f32>,
        att_scores: &mut Tensor<f32>,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        seq_len: usize,
        total_seq_len: usize,
    ) {
        let dqkv = self.d / self.config.num_attention_heads;
        let sqrt_dqkv = (dqkv as f32).sqrt();

        // 计算注意力分数
        for h in 0..self.n_kv_h {
            for g in 0..(self.config.num_attention_heads / self.n_kv_h) {
                for i in 0..seq_len {
                    for j in 0..total_seq_len {
                        let mut score = 0.0;
                        for k_idx in 0..dqkv {
                            score += q.data()[i * self.config.num_attention_heads * dqkv
                                + (h * (self.config.num_attention_heads / self.n_kv_h) + g) * dqkv
                                + k_idx]
                                * k.data()[j * self.n_kv_h * dqkv + h * dqkv + k_idx];
                        }
                        unsafe {
                            att_scores.data_mut()[h
                                * (self.config.num_attention_heads / self.n_kv_h)
                                * seq_len
                                * total_seq_len
                                + g * seq_len * total_seq_len
                                + i * total_seq_len
                                + j] = score / sqrt_dqkv;
                        }
                    }
                }
            }
        }

        // 应用 softmax
        for h in 0..self.n_kv_h {
            for g in 0..(self.config.num_attention_heads / self.n_kv_h) {
                for i in 0..seq_len {
                    let mut max_score = f32::NEG_INFINITY;
                    for j in 0..total_seq_len {
                        let score = att_scores.data()[h
                            * (self.config.num_attention_heads / self.n_kv_h)
                            * seq_len
                            * total_seq_len
                            + g * seq_len * total_seq_len
                            + i * total_seq_len
                            + j];
                        max_score = max_score.max(score);
                    }

                    let mut sum_exp = 0.0;
                    for j in 0..total_seq_len {
                        unsafe {
                            let score = att_scores.data()[h
                                * (self.config.num_attention_heads / self.n_kv_h)
                                * seq_len
                                * total_seq_len
                                + g * seq_len * total_seq_len
                                + i * total_seq_len
                                + j];
                            let exp_score = (score - max_score).exp();
                            att_scores.data_mut()[h
                                * (self.config.num_attention_heads / self.n_kv_h)
                                * seq_len
                                * total_seq_len
                                + g * seq_len * total_seq_len
                                + i * total_seq_len
                                + j] = exp_score;
                            sum_exp += exp_score;
                        }
                    }

                    for j in 0..total_seq_len {
                        unsafe {
                            att_scores.data_mut()[h
                                * (self.config.num_attention_heads / self.n_kv_h)
                                * seq_len
                                * total_seq_len
                                + g * seq_len * total_seq_len
                                + i * total_seq_len
                                + j] /= sum_exp;
                        }
                    }
                }
            }
        }

        // 计算输出
        let mut output = Tensor::<f32>::default(&[seq_len, self.d]);
        for h in 0..self.n_kv_h {
            for g in 0..(self.config.num_attention_heads / self.n_kv_h) {
                for i in 0..seq_len {
                    for j in 0..total_seq_len {
                        unsafe {
                            let att_score = att_scores.data()[h
                                * (self.config.num_attention_heads / self.n_kv_h)
                                * seq_len
                                * total_seq_len
                                + g * seq_len * total_seq_len
                                + i * total_seq_len
                                + j];
                            for k_idx in 0..dqkv {
                                output.data_mut()[i * self.d
                                    + (h * (self.config.num_attention_heads / self.n_kv_h) + g)
                                        * dqkv
                                    + k_idx] +=
                                    att_score * v.data()[j * self.n_kv_h * dqkv + h * dqkv + k_idx];
                            }
                        }
                    }
                }
            }
        }

        // 更新 hidden_states
        unsafe {
            let output_data = output.data();
            let hidden_states_data = hidden_states.data_mut();
            hidden_states_data[..(seq_len * self.d)]
                .copy_from_slice(&output_data[..(seq_len * self.d)]);
        }
    }

    fn mlp(
        &self,
        residual: &mut Tensor<f32>,
        hidden_states: &mut Tensor<f32>,
        gate: &mut Tensor<f32>,
        up: &mut Tensor<f32>,
        layer: usize,
    ) {
        let seq_len = hidden_states.size() / self.d;

        OP::rms_norm(
            hidden_states,
            residual,
            &self.params.rms_ffn_w[layer],
            self.eps,
        );

        OP::matmul_transb(gate, 0., hidden_states, &self.params.w_gate[layer], 1.0);
        OP::matmul_transb(up, 0., hidden_states, &self.params.w_up[layer], 1.0);

        // SiLU activation
        unsafe {
            for i in 0..seq_len * self.config.intermediate_size {
                gate.data_mut()[i] = gate.data()[i] / (1.0 + (-gate.data()[i]).exp());
            }
            for i in 0..seq_len * self.config.intermediate_size {
                up.data_mut()[i] *= gate.data()[i];
            }
        }

        let mut out = Tensor::<f32>::default(&[seq_len, self.d]);
        OP::matmul_transb(&mut out, 0., up, &self.params.w_down[layer], 1.0);

        // 添加残差连接
        unsafe {
            let residual_data = residual.data_mut();
            let out_data = out.data();
            for i in 0..seq_len * self.d {
                residual_data[i] += out_data[i];
            }
        }
    }

    pub fn clear_history(&mut self) {
        self.history = ChatHistory::new();
        self.kv_cache = KVCache::new(
            self.n_layers,
            self.max_seq_len,
            self.n_kv_h * (self.d / self.config.num_attention_heads),
            0,
        );
    }
}
