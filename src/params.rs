use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
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

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // for name in safetensor.names() {
        //     println!("{}", name);
        // }
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let data = tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Tensor::<f32>::new(data, tensor.shape())
        };

        // 初始化存储层参数的向量
        let num_layers = config.num_hidden_layers;
        let load_layer_params = |prefix: &str| -> Vec<Tensor<f32>> {
            (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.{}", i, prefix)))
                .collect()
        };

        LLamaParams {
            // embedding_table: get_tensor("lm_head.weight"),
            embedding_table: get_tensor(if config.tie_word_embeddings {
                "lm_head.weight"
            } else {
                "model.embed_tokens.weight"
            }),

            // decoder layer
            rms_att_w: load_layer_params("input_layernorm.weight"),
            wq: load_layer_params("self_attn.q_proj.weight"),
            wk: load_layer_params("self_attn.k_proj.weight"),
            wv: load_layer_params("self_attn.v_proj.weight"),
            wo: load_layer_params("self_attn.o_proj.weight"),
            // ffn layer
            rms_ffn_w: load_layer_params("post_attention_layernorm.weight"),
            w_up: load_layer_params("mlp.up_proj.weight"),
            w_gate: load_layer_params("mlp.gate_proj.weight"),
            w_down: load_layer_params("mlp.down_proj.weight"),
            // output
            rms_out_w: get_tensor("model.norm.weight"), // (hidden_size, )
            lm_head: get_tensor("lm_head.weight"),      // (vocab_size, dim)
        }
    }
}
