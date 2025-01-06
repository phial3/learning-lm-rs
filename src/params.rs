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
        // 首先打印所有可用的张量名称，帮助调试
        println!("Available tensors:");
        for name in safetensor.names() {
            println!("  {}", name);
        }

        // 辅助函数：从safetensors中加载张量
        let get_tensor = |name: &str| -> Tensor<f32> {
            // 获取张量数据和形状
            let view = safetensor.tensor(name).unwrap_or_else(|_| {
                panic!("Failed to load tensor: {}", name)
            });
            let shape = view.shape().to_vec();

            // 获取数据
            assert_eq!(view.dtype(), safetensors::Dtype::F32, "Only F32 dtype is supported for tensor: {}", name);

            let data: Vec<f32> = view.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();

            Tensor::new(data, &shape)
        };

        // 初始化存储层参数的向量
        let num_layers = config.num_hidden_layers;
        let mut rms_att_w = Vec::with_capacity(num_layers);
        let mut wq = Vec::with_capacity(num_layers);
        let mut wk = Vec::with_capacity(num_layers);
        let mut wv = Vec::with_capacity(num_layers);
        let mut wo = Vec::with_capacity(num_layers);
        let mut rms_ffn_w = Vec::with_capacity(num_layers);
        let mut w_up = Vec::with_capacity(num_layers);
        let mut w_gate = Vec::with_capacity(num_layers);
        let mut w_down = Vec::with_capacity(num_layers);

        // 加载每一层的参数
        for layer_idx in 0..num_layers {
            // Attention层参数
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", layer_idx)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx)));

            // FFN层参数
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", layer_idx)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx)));
        }

        // 加载输出层参数
        let rms_out_w = get_tensor("model.norm.weight");

        // 加载 lm_head 权重，当 tie_word_embeddings 为 true 时也用作 embedding 表
        let lm_head = get_tensor("lm_head.weight");

        let embedding_table = if config.tie_word_embeddings {
            // 如果共享权重，使用 lm_head 权重
            lm_head.slice(0, lm_head.shape())
        } else {
            // 这种情况应该不会发生，因为知道模型使用权重共享
            panic!("Model requires separate embedding table but it's not present in the safetensors file")
        };

        LLamaParams {
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
