use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use half::{bf16, f16};
use safetensors::SafeTensors;
use std::mem;

pub trait FromLeBytes: Sized {
    fn from_le_bytes(bytes: &[u8]) -> Self;
}
impl FromLeBytes for f32 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().expect("Invalid byte length for f32"))
    }
}
impl FromLeBytes for f16 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f16::from_le_bytes([bytes[0], bytes[1]])
    }
}
impl FromLeBytes for bf16 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bf16::from_le_bytes([bytes[0], bytes[1]])
    }
}
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

impl<T: Copy + Default + FromLeBytes + 'static> LLamaParams<T> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<T> {
            let tensor = safetensor
                .tensor(name)
                .unwrap_or_else(|_| panic!("Tensor {} not found", name));
            let tensor_dtype = tensor.dtype();
            let element_size = match tensor_dtype {
                safetensors::Dtype::F32 => mem::size_of::<f32>(),
                safetensors::Dtype::F16 => mem::size_of::<f16>(),
                _ => panic!("Unsupported data type: {:?}", tensor_dtype),
            };

            let data_bytes = tensor.data();
            let shape = tensor.shape().to_vec();
            // 将字节数组解析为指定的数据类型
            let data = data_bytes
                .chunks_exact(element_size)
                .map(|chunk| T::from_le_bytes(chunk))
                .collect::<Vec<T>>();

            Tensor::new(data, &shape)
        };

        let lm_head = get_tensor("lm_head.weight");

        // 处理共享参数：通过复制数据创建新Tensor
        let embedding_table = if config.tie_word_embeddings {
            let data = lm_head.data().to_vec(); // 复制数据
            let shape = lm_head.shape().clone();
            Tensor::new(data, &shape)
        } else {
            get_tensor("model.embed_tokens.weight")
        };

        // 加载各层参数（与原代码一致）
        let num_hidden_layers = config.num_hidden_layers;
        let mut rms_att_w = Vec::with_capacity(num_hidden_layers);
        let mut wq = Vec::with_capacity(num_hidden_layers);
        let mut wk = Vec::with_capacity(num_hidden_layers);
        let mut wv = Vec::with_capacity(num_hidden_layers);
        let mut wo = Vec::with_capacity(num_hidden_layers);
        let mut rms_ffn_w = Vec::with_capacity(num_hidden_layers);
        let mut w_up = Vec::with_capacity(num_hidden_layers);
        let mut w_gate = Vec::with_capacity(num_hidden_layers);
        let mut w_down = Vec::with_capacity(num_hidden_layers);

        for layer_idx in 0..num_hidden_layers {
            rms_att_w.push(get_tensor(&format!(
                "model.layers.{}.input_layernorm.weight",
                layer_idx
            )));
            wq.push(get_tensor(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            )));
            wk.push(get_tensor(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            )));
            wv.push(get_tensor(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            )));
            wo.push(get_tensor(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            )));
            rms_ffn_w.push(get_tensor(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                layer_idx
            )));
            w_up.push(get_tensor(&format!(
                "model.layers.{}.mlp.up_proj.weight",
                layer_idx
            )));
            w_gate.push(get_tensor(&format!(
                "model.layers.{}.mlp.gate_proj.weight",
                layer_idx
            )));
            w_down.push(get_tensor(&format!(
                "model.layers.{}.mlp.down_proj.weight",
                layer_idx
            )));
        }

        let rms_out_w = get_tensor("model.norm.weight");

        Self {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}
