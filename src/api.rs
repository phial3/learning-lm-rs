use crate::chat::Chat;
use crate::config::LlamaConfigJson;
use crate::model;
use crate::params::ChatParams;
use actix_cors::Cors;
use actix_web::{get, middleware, post, web, App, HttpResponse, HttpServer, Responder};
use log::{error, info};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::Tokenizer;

// 全局状态，保存模型实例
struct AppState {
    story_model: Mutex<Option<(model::Llama<f32>, Tokenizer)>>,
    chat_model: Mutex<Option<Chat>>,
}

// 请求和响应结构体
#[derive(Deserialize)]
struct StoryRequest {
    prompt: String,
    max_length: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    system_message: Option<String>,
    clear_history: Option<bool>,
}

#[derive(Serialize)]
struct ApiResponse {
    status: String,
    data: String,
}

// API状态检查
#[get("/status")]
async fn status() -> impl Responder {
    HttpResponse::Ok().json(ApiResponse {
        status: "success".to_string(),
        data: "API服务正常运行".to_string(),
    })
}

// Story生成API
#[post("/story")]
async fn generate_story(req: web::Json<StoryRequest>, data: web::Data<AppState>) -> impl Responder {
    let mut story_model_guard = data.story_model.lock().unwrap();

    // 如果模型未加载，则加载模型
    if story_model_guard.is_none() {
        info!("首次请求，加载story模型...");
        match load_story_model() {
            Ok((model, tokenizer)) => {
                *story_model_guard = Some((model, tokenizer));
            }
            Err(e) => {
                error!("加载story模型失败: {}", e);
                return HttpResponse::InternalServerError().json(ApiResponse {
                    status: "error".to_string(),
                    data: format!("加载模型失败: {}", e),
                });
            }
        }
    }

    if let Some((model, tokenizer)) = story_model_guard.as_mut() {
        let max_length = req.max_length.unwrap_or(500);
        let temperature = req.temperature.unwrap_or(0.8);
        let top_p = req.top_p.unwrap_or(0.95);
        let top_k = req.top_k.unwrap_or(40);

        let tokens = tokenizer.encode(req.prompt.as_str(), true).unwrap();
        let input_ids = tokens.get_ids();

        let output_ids = model.generate(input_ids, max_length, top_p, top_k, temperature);

        let generated_text = tokenizer
            .decode(&output_ids[input_ids.len()..], true)
            .unwrap();

        HttpResponse::Ok().json(ApiResponse {
            status: "success".to_string(),
            data: generated_text,
        })
    } else {
        HttpResponse::InternalServerError().json(ApiResponse {
            status: "error".to_string(),
            data: "模型加载失败".to_string(),
        })
    }
}

// Chat API
#[post("/chat")]
async fn generate_chat(req: web::Json<ChatRequest>, data: web::Data<AppState>) -> impl Responder {
    let mut chat_model_guard = data.chat_model.lock().unwrap();

    // 如果模型未加载，则加载模型
    if chat_model_guard.is_none() {
        info!("首次请求，加载chat模型...");
        match load_chat_model() {
            Ok(chat) => {
                *chat_model_guard = Some(chat);
            }
            Err(e) => {
                error!("加载chat模型失败: {}", e);
                return HttpResponse::InternalServerError().json(ApiResponse {
                    status: "error".to_string(),
                    data: format!("加载模型失败: {}", e),
                });
            }
        }
    }

    if let Some(chat) = chat_model_guard.as_mut() {
        // 如果提供了系统消息，则设置系统消息
        if let Some(system_message) = &req.system_message {
            chat.set_system_message(system_message.clone());
        }

        // 如果需要清除历史记录
        if req.clear_history.unwrap_or(false) {
            chat.clear_history();
        }

        // 生成回复
        let response = chat.chat(&req.message);

        HttpResponse::Ok().json(ApiResponse {
            status: "success".to_string(),
            data: response,
        })
    } else {
        HttpResponse::InternalServerError().json(ApiResponse {
            status: "error".to_string(),
            data: "模型加载失败".to_string(),
        })
    }
}

// 加载story模型
fn load_story_model() -> Result<(model::Llama<f32>, Tokenizer), String> {
    info!("加载story模型...");
    let model_dir = PathBuf::from("models").join("story");

    if !model_dir.exists() {
        return Err(format!("模型目录不存在: {:?}", model_dir));
    }

    let model = model::Llama::<f32>::from_safetensors(&model_dir);

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(format!("Tokenizer文件不存在: {:?}", tokenizer_path));
    }

    let tokenizer = match Tokenizer::from_file(tokenizer_path) {
        Ok(t) => t,
        Err(e) => return Err(format!("加载tokenizer失败: {}", e)),
    };

    Ok((model, tokenizer))
}

// 加载chat模型
fn load_chat_model() -> Result<Chat, String> {
    info!("加载chat模型...");
    let config_path = "models/chat/config.json";
    let tokenizer_path = "models/chat/tokenizer.json";
    let model_path = "models/chat/model.safetensors";

    // 检查文件是否存在
    if !std::path::Path::new(config_path).exists() {
        return Err(format!("配置文件不存在: {}", config_path));
    }
    if !std::path::Path::new(tokenizer_path).exists() {
        return Err(format!("Tokenizer文件不存在: {}", tokenizer_path));
    }
    if !std::path::Path::new(model_path).exists() {
        return Err(format!("模型文件不存在: {}", model_path));
    }

    // 加载配置
    let config = LlamaConfigJson::from_file(config_path.into());

    // 加载tokenizer
    let tokenizer = match Tokenizer::from_file(tokenizer_path) {
        Ok(t) => t,
        Err(e) => return Err(format!("加载tokenizer失败: {}", e)),
    };

    // 加载模型参数
    let model_data = match fs::read(model_path) {
        Ok(data) => data,
        Err(e) => return Err(format!("读取模型文件失败: {}", e)),
    };

    let safetensor = match SafeTensors::deserialize(&model_data) {
        Ok(t) => t,
        Err(e) => return Err(format!("反序列化模型失败: {}", e)),
    };

    let params = ChatParams::from_safetensors(&safetensor, &config);
    let chat = Chat::new(params, tokenizer, config);

    Ok(chat)
}

// 启动API服务
pub fn start_api_server() -> std::io::Result<()> {
    // 初始化日志
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    // 创建应用状态
    let app_state = web::Data::new(AppState {
        story_model: Mutex::new(None),
        chat_model: Mutex::new(None),
    });

    info!("启动API服务在 http://127.0.0.1:8000");

    // 创建HTTP服务器
    let server = HttpServer::new(move || {
        // 配置CORS
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(middleware::Logger::default())
            .wrap(cors)
            .app_data(app_state.clone())
            .service(status)
            .service(generate_story)
            .service(generate_chat)
    })
    .bind("127.0.0.1:8000")?;

    println!("服务器启动在 http://127.0.0.1:8000");

    // 使用actix_web::rt::System来运行异步任务
    use actix_web::rt::System;
    let sys = System::new();
    sys.block_on(server.run())?;

    Ok(())
}
