use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use half::{bf16, f16};
use safetensors::SafeTensors;

#[derive(Clone)]
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

pub trait FromBytes: Sized {
    fn from_bytes(bytes: &[u8]) -> Vec<Self>;
}

impl FromBytes for f32 {
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytemuck::cast_slice(bytes).to_vec()
    }
}
impl FromBytes for f16 {
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        let expected = std::mem::size_of::<f16>();
        bytes
            .chunks_exact(expected)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }
}
impl FromBytes for bf16 {
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        let expected = std::mem::size_of::<bf16>();
        bytes
            .chunks_exact(expected)
            .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }
}

impl<T: Copy + Clone + Default + FromBytes> LLamaParams<T> {
    pub(crate) fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<T> {
            match safetensor.tensor(name) {
                Ok(v) => {
                    let data: Vec<T> = T::from_bytes(v.data());

                    Tensor::new(data, &v.shape().to_vec())
                }
                Err(_) => {
                    panic!("Failed to load tensor: {}", name);
                }
            }
        };

        let layers = config.num_hidden_layers;
        let embedding_table = if config.tie_word_embeddings {
            get_tensor("lm_head.weight")
        } else {
            get_tensor("model.embed_tokens.weight")
        };
        Self {
            embedding_table: embedding_table,
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
