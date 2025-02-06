use learning_lm_rust::chat;
#[cfg(not(feature = "cuda"))]
use learning_lm_rust::model::{self as model};
#[cfg(feature = "cuda")]
use learning_lm_rust::model_cuda::{self as model};
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut cache = llama.new_cache();
    let mut session = chat::ChatSession::new();
    println!("欢迎使用聊天助手！输入 'quit' 结束对话。");

    session.add_message("system", "You are a helpful assistant");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "quit" {
            break;
        }

        session.add_message("user", input);

        let prompt = session.format_prompt();
        let response = llama.chat(&prompt, &mut cache, &tokenizer, 500, 0.8, 30, 1.);

        println!("Asistant: {}", response);
        session.add_message("assistant", &response);
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use tokenizers::Tokenizer;

    #[test]
    fn test_tokenizer() -> Result<(), Box<dyn std::error::Error>> {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

        let system_message = "你是一个有用的助手。";
        let user_message = "你好，今天天气怎么样？";

        let chat_input = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant",
            system_message, user_message
        );

        let encoding = tokenizer.encode(chat_input, false).unwrap();
        let token_ids = encoding.get_ids();

        assert_eq!(
            token_ids.to_vec(),
            vec![
                32001, 1587, 13, 29383, 28971, 28518, 28998, 28963, 28914, 30278, 29427, 28944,
                32000, 28705, 13, 32001, 2188, 13, 29383, 29530, 28924, 30316, 29354, 29354, 30627,
                31401, 29797, 29675, 29771, 32000, 28705, 13, 32001, 13892
            ]
        );
        Ok(())
    }
}
