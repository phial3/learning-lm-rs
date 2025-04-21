#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct LlamaConfigJson {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub torch_dtype: String,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub sliding_window: usize,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub initializer_range: f32,
}

impl LlamaConfigJson {
    pub fn from_file(path: std::path::PathBuf) -> Self {
        let config_str = std::fs::read_to_string(path).expect("Failed to read config file");
        let mut config: Self =
            serde_json::from_str(&config_str).expect("Failed to parse config file");

        // 设置默认值
        if config.rms_norm_eps == 0.0 {
            config.rms_norm_eps = default_rms_norm_eps();
        }
        if config.rope_theta == 0.0 {
            config.rope_theta = default_rope_theta();
        }
        if config.sliding_window == 0 {
            config.sliding_window = 1024;
        }
        if config.hidden_act.is_empty() {
            config.hidden_act = "silu".to_string();
        }
        if config.attention_dropout == 0.0 {
            config.attention_dropout = 0.0;
        }
        if config.initializer_range == 0.0 {
            config.initializer_range = 0.02;
        }

        config
    }
}

#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}

#[inline(always)]
const fn default_tie_word_embeddings() -> bool {
    false
}
