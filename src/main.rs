mod api;
mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use crate::model::Llama;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// 根据不同的菜单选项运行不同的功能
fn main() -> std::io::Result<()> {
    println!("Welcome to Llama!");
    println!("1. Generate story");
    println!("2. Chat with the model, type 'exit' to quit");
    println!("3. Run chat API server");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let choice = line.trim().parse().unwrap();
    match choice {
        1 => story(),
        2 => chat_story(),
        3 => chat_api_server(),
        _ => panic!("Invalid choice"),
    }
}

fn story() -> std::io::Result<()> {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1.);
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());

    Ok(())
}

/// chat with the model
fn chat_story() -> std::io::Result<()> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/story");
    let model = Llama::<f32>::from_safetensors(&model_path);
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap();
    let mut cache = model.new_cache();
    let mut history = Vec::new();

    loop {
        println!("User: ");
        let mut user_input = String::new();
        std::io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();

        if user_input == "exit" {
            break;
        }

        let response = model.chat(user_input, &tokenizer, &mut history, &mut cache);
        println!("AI: {}", response);
    }

    Ok(())
}

/// Chat API Server
#[actix_web::main]
async fn chat_api_server() -> std::io::Result<()> {
    let model_dir = std::path::Path::new("models/story");
    let model = Arc::new(Llama::from_safetensors(model_dir));

    let tokenizer = Arc::new(
        Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("Failed to load tokenizer"),
    );

    println!("Server running on http://localhost:8080");
    api::run_api_server(model, tokenizer, 8080).await
}
