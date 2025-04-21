use crate::chat::ChatEngine;
use crate::model;
use axum::{
    extract::{Json, State},
    response::IntoResponse,
    routing::post,
    Router,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

// Add serde's Deserialize derive
#[derive(serde::Deserialize)]
struct ChatRequest {
    user_id: String,
    user_input: String,
}
#[derive(serde::Deserialize)]
struct StoryRequest {
    user_input: String,
}
#[derive(Clone)]
struct AppState {
    bot: Arc<ChatEngine>,
}
#[derive(serde::Serialize)]
struct Response {
    output: String,
}
// curl -X POST http://localhost:3000/chat \
//   -H "Content-Type: application/json" \
//   -d '{"user_id": "marklee" ,"user_input": "Tell me a story"}'
async fn chat_app(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> impl IntoResponse {
    // Chat app logic
    let bot = state.bot.clone();
    let output = bot.generate(payload.user_id, payload.user_input);

    Json(Response { output })
}
// curl -X POST http://localhost:3000/story \
//   -H "Content-Type: application/json" \
//   -d '{"user_input": "Once upon a time"}'
async fn story_app(Json(payload): Json<StoryRequest>) -> impl IntoResponse {
    // format!("Response for user: {}", payload.user_id)
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // 在这里可以定义有关故事模型的特定操作
    let input = payload.user_input.as_str();
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("{}", input);

    // 调用生成函数
    let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1.);

    let output = payload.user_input + tokenizer.decode(&output_ids, true).unwrap().as_str();

    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    Json(Response { output })
}

pub async fn start_server() {
    println!("🦀Starting Chat Server...🥳");
    let state = Arc::new(AppState {
        bot: Arc::new(ChatEngine::new()),
    });

    let app = Router::new()
        .route("/chat", post(chat_app))
        .route("/story", post(story_app))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
