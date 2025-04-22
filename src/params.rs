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
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let data = tensor.data();

            // let mut data_vec = vec![0.0; len]; 不可强制转换

            //创建一个向量，同时预先为其分配指定大小的内存空间。
            //一个 f32 类型的数据在内存中占用 4 个字节，而 data 是存储字节数据的切片，所以 data.len() / 4 计算出的就是 data 里能够转换得到的 f32 元素的数量。
            let mut values: Vec<f32> = Vec::with_capacity(data.len() / 4);
            // 将 4 个字节的数组转换为 f32 类型的函数
            for chunk in data.chunks_exact(4) {
                values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Tensor::<f32>::new(values, tensor.shape())
        };

        let num_layers = config.num_hidden_layers;
        let _num_heads = config.num_attention_heads;

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            lm_head: get_tensor("lm_head.weight"),
            // lm_head: if config.tie_word_embeddings {
            //     get_tensor("embedding_tokens.weight")  // 复用 embedding_table
            // } else {
            //     get_tensor("lm_head.weight")
            // },
            rms_att_w: (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
                .collect(),
            rms_ffn_w: (0..num_layers)
                .map(|i| {
                    get_tensor(&format!(
                        "model.layers.{}.post_attention_layernorm.weight",
                        i
                    ))
                })
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            w_down: (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
                .collect(),
            w_up: (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
                .collect(),
            w_gate: (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
                .collect(),
            wq: (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
                .collect(),
            wk: (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
                .collect(),
            wv: (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
                .collect(),
            wo: (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
                .collect(),
        }
    }
}
