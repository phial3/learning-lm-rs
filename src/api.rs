// src/api.rs
use crate::Llama;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;

// 请求体结构（JSON反序列化）
#[derive(Deserialize)]
struct GenerateRequest {
    text: String,
    max_length: Option<usize>,
    temperature: Option<f32>,
}

// 响应体结构（JSON序列化）
#[derive(Serialize)]
struct GenerateResponse {
    generated_text: String,
    latency_ms: u64,
}

// 共享状态结构
struct AppState {
    model: Arc<Llama<f32>>,
    tokenizer: Arc<Tokenizer>,
}

/// 生成端点实现
/// 处理POST请求，接收JSON请求体，返回JSON响应体
///
/// ```bash
/// curl -X POST http://localhost:8080/generate \
/// "Content-Type: application/json" \
/// '{
///     "text": "an Apple",
///     "max_length": 300,
///     "temperature": 0.8
/// }'
/// ```
///
async fn generate(data: web::Data<AppState>, req: web::Json<GenerateRequest>) -> impl Responder {
    let start_time = std::time::Instant::now();

    // 编码输入
    // let encoding = data.tokenizer.encode(&req.text, true).unwrap();
    let encoding = data.tokenizer.encode(req.text.as_str(), true).unwrap();
    let input_ids = encoding.get_ids();

    // 执行推理
    let output_ids = data.model.generate(
        input_ids,
        req.max_length.unwrap_or(100),
        0.8, // top_p
        50,  // top_k
        req.temperature.unwrap_or(0.7),
    );

    // 解码输出
    let generated_text = data.tokenizer.decode(&output_ids, true).unwrap();

    HttpResponse::Ok().json(GenerateResponse {
        generated_text,
        latency_ms: start_time.elapsed().as_millis() as u64,
    })
}

/// 启动API服务
pub async fn run_api_server(
    model: Arc<Llama<f32>>,
    tokenizer: Arc<Tokenizer>,
    port: u16,
) -> std::io::Result<()> {
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(AppState {
                model: Arc::clone(&model),
                tokenizer: Arc::clone(&tokenizer),
            }))
            .wrap(
                actix_cors::Cors::default()
                    .allow_any_origin()
                    .allowed_methods(vec!["POST"])
                    .max_age(3600),
            )
            .service(web::resource("/generate").route(web::post().to(generate)))
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
