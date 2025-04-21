use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use half::f16;
use safetensors::{tensor, SafeTensors};
use std::slice;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers 注意很多权重是以转制的形式存储的
    pub wq: Vec<Tensor<T>>, // (n_heads * head_size, hidden_size) x layers 注意这里Q的头可能是KV的整数倍, 因为用的是GHQ
    pub wk: Vec<Tensor<T>>, // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>, // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>, // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let layers = config.num_hidden_layers;
        let get_tensor = |name: &str| {
            match safetensor.tensor(name) {
                Ok(tensor_view) => {
                    // 注意这里获取length和转换为f32的方式
                    // 这里是真的不会，对rust不熟，参考了一个大佬的实现
                    let p: usize = tensor_view.shape().iter().product();
                    let new_data = unsafe {
                        slice::from_raw_parts(tensor_view.data().as_ptr() as *const f32, p)
                    };
                    Tensor::new(Vec::from(new_data), &tensor_view.shape().to_vec())
                }
                Err(err) => panic!("{}", err),
            }
        };

        LLamaParams {
            // tie_word_embeddings = true
            // 注意，如果tie_word_embeddings为true，那么embedding_table和lm_head是同一个
            // chat模型里不是共用的
            embedding_table: if config.tie_word_embeddings {
                get_tensor("lm_head.weight")
            } else {
                get_tensor("model.embed_tokens.weight") // chat模型中的
            },

            rms_att_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")))
                .collect(),
            wq: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .collect(),

            rms_ffn_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .collect(),

            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}

impl LLamaParams<f16> {
    #[allow(unstable_features)]
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let layers = config.num_hidden_layers;
        let get_tensor = |name: &str| {
            match safetensor.tensor(name) {
                Ok(tensor_view) => {
                    // 注意这里获取length和转换为f16的方式
                    let p: usize = tensor_view.shape().iter().product();
                    let new_data = unsafe {
                        slice::from_raw_parts(tensor_view.data().as_ptr() as *const f16, p)
                    };
                    Tensor::new(Vec::from(new_data), &tensor_view.shape().to_vec())
                }
                Err(err) => panic!("{}", err),
            }
        };

        LLamaParams {
            embedding_table: if config.tie_word_embeddings {
                get_tensor("lm_head.weight")
            } else {
                get_tensor("model.embed_tokens.weight") // chat模型中的
            },

            rms_att_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")))
                .collect(),
            wq: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .collect(),

            rms_ffn_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .collect(),

            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
