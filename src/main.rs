mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use crate::config::LlamaConfigJson;
use dashmap::DashMap;
use half::bf16;
use operators::ToF32;
use params::Load;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::sync::Arc;
use std::{f32, path::PathBuf};

use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use tokenizers::Tokenizer;

#[derive(Serialize, Deserialize, Debug)]
struct Request {
    session_id: String,
    history: String,
    system_message: String,
    user_message: String,
}

#[get("/story")]
async fn story() -> impl Responder {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    let output_ids = llama.generate(input_ids, 200, 0.8, 30, 0.6);
    let mut ans = tokenizer.decode(&output_ids, true).unwrap();
    ans.insert_str(0, input);
    HttpResponse::Ok().body(ans)
}

fn chat_func<T>(model_dir: PathBuf, prompt: &Request) -> String
where
    T: Default + Copy + Load + ToF32,
{
    let llama = model::Llama::<T>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = format!("{0}<|im_start|>system\n{1}<|im_end|>\n<|im_start|>user\n{2}<|im_end|>\n<|im_start|>assistant",
                        prompt.history,prompt.system_message,prompt.user_message);
    println!("{}\n", "-".repeat(50));
    // println!("{}\n{}",(|| "-".repeat(50))(),&input);
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    let output_ids = llama.generate(input_ids, 100, 0.8, 30, 1.);
    tokenizer.decode(&output_ids, true).unwrap()
}

#[post("/chat")]
async fn chat(
    mut prompt_json: web::Json<Request>,
    app_data: web::Data<Arc<DashMap<String, String>>>,
) -> impl Responder {
    println!(
        "\n{}\nreceive request from session_id = {{{}}}\n{:?}",
        "-".repeat(50),
        &prompt_json.session_id,
        &prompt_json
    );
    let map = app_data.as_ref();
    let history = map
        .get(&prompt_json.session_id)
        .map(|v| v.to_string())
        .unwrap_or_default();
    prompt_json.history = history;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let config = File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
    let ans = match config.torch_dtype.as_ref() {
        "bfloat16" => chat_func::<bf16>(model_dir, &prompt_json),
        "float32" => chat_func::<f32>(model_dir, &prompt_json),
        _ => todo!(),
    };
    map.insert(prompt_json.session_id.clone(),format!("{0}<|im_start|>system\n{1}<|im_end|>\n<|im_start|>user\n{2}<|im_end|>\n<|im_start|>assistant{3}\n",prompt_json.history,prompt_json.system_message,prompt_json.user_message,ans.clone()));
    HttpResponse::Ok().body(ans)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let app_data = web::Data::new(Arc::new(DashMap::<String, String>::new()));
    println!("Server running on http://127.0.0.1:8080");
    HttpServer::new(move || {
        App::new()
            .app_data(app_data.clone())
            .service(story)
            .service(chat)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

#[test]
fn infer_test() {
    let dir = "chat";
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join(dir);
    let config = File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
    let prompt_json = Request {
        session_id: "".to_string(),
        history: "".to_string(),
        system_message: "you are a helpful assistant".to_string(),
        user_message: "who are you?".to_string(),
    };
    let ans = match config.torch_dtype.as_ref() {
        "bfloat16" => chat_func::<bf16>(model_dir, &prompt_json),
        "float32" => chat_func::<f32>(model_dir, &prompt_json),
        _ => todo!(),
    };
    println!("{}", ans);
}
