use crate::kvcache::KVCache;
use crate::model::Llama;
use dashmap::DashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;
struct Conversation {
    input: String,
    output: String,
}
struct Session {
    // 会话标题
    title: String,
    // 会话历史
    history: Vec<(usize, Conversation)>,
    // 会话缓存
    kvcache: KVCache<f32>,
}
pub struct ChatEngine {
    // 模型参数
    model: Llama<f32>,
    tokenizer: Tokenizer,
    sessions: DashMap<String, Session>, // ✅ 并发安全
    // 聊天模型特有字段
    system_prompt: String,
}
pub struct ChatModel {
    // 模型参数
    model: Llama<f32>,
    tokenizer: Tokenizer,

    sessions: Vec<Session>,
    // 聊天模型特有字段
    system_prompt: String,
    // ...
}

/// 聊天模型的实现
///         外部接口
///             1. 实现ChatModel::new方法，用于初始化聊天模型
///             2. generator trait实现
///             3. 实现ChatModel::process_history方法，用于处理对话历史，会话回滚
///             4. 实现会话切换，多会话管理
///         内部接口
///             1. ChatModel::format_prompt方法，用于格式化用户输入
impl ChatModel {
    pub fn new() -> Self {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");

        // 先创建临时model实例
        let model = Llama::<f32>::from_safetensors(&model_dir);

        // 现在可以安全初始化结构体字段
        ChatModel {
            model,
            tokenizer: Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap(),
            sessions: Vec::new(),
            system_prompt: "You are a chatbot".to_string(),
        }
    }
    pub fn begin(&mut self) {
        // let mut sessions: HashMap<String, Session> = HashMap::new();

        println!("Welcome to the chatbot!");
        loop {
            println!("Choose an option:");
            println!("1. Start a new conversation");
            println!("2. Continue a previous conversation");
            println!("3. Exit");
            print!("Enter your choice : ");
            io::stdout().flush().unwrap();

            let mut choice = String::new();
            io::stdin().read_line(&mut choice).unwrap();
            let choice = choice.trim();

            match choice {
                "1" => {
                    // 开启新会话
                    self.new_session();
                }
                "2" => {
                    // 继续之前的会话
                    self.history_session();
                }
                "3" => {
                    println!("Goodbye!");
                    break;
                }
                _ => {
                    println!("Invalid choice. Please choose again.");
                }
            }
        }
    }

    fn generator(&mut self, session_id: usize) {
        {
            let session = self.sessions.get_mut(session_id).unwrap();

            // 显示欢迎信息
            if session.history.is_empty() {
                println!("-----------------------------------------");
                println!("                欢迎！                  ");
                println!("请输入任意内容与我聊天，或输入以下命令：");
                println!("  - 'exit'        ：退出程序");
                println!("  - 'rollback'    ：撤回上一次对话");
                println!("-----------------------------------------");
                println!("Hello, I am a chatbot");
            } else {
                println!("-----------------------------------------");
                println!("欢迎回来！以下是之前的对话记录：");
                for (i, (_, conversation)) in session.history.iter().enumerate() {
                    println!("对话 {}:", i + 1);
                    println!("User: {}", conversation.input);
                    println!("Assistant: {}", conversation.output);
                }
                println!("-----------------------------------------");
                println!("Welcome back to the conversation");
            }
        }
        loop {
            print!("User: ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            if input.trim() == "exit" {
                println!("Goodbye!");
                break;
            } else if input.trim() == "rollback" {
                println!("Rolling back to the previous conversation...");

                self.roll_back(session_id);
                // 回滚到上一次的对话
                // ...
                continue;
            }

            let format_input = format_prompt(input.clone());
            let session = self.sessions.get_mut(session_id).unwrap();
            if session.history.is_empty() {
                session.title = input.clone();
                // format_input = format!("<|im_start|>system\n{}<|im_end|>\n{}",self.system_prompt, format_input);
            }

            let binding = self.tokenizer.encode(format_input.clone(), true).unwrap();
            let input_ids = binding.get_ids();

            // 调用生成函数
            let output_ids =
                self.model
                    .chat_generator(input_ids, 300, 0.8, 20, 1., &mut session.kvcache);

            let output = self.tokenizer.decode(&output_ids, true).unwrap();
            let conversation = Conversation {
                input: input.clone(),
                output: output.clone(),
            };
            session
                .history
                .push((input_ids.len() + output_ids.len(), conversation));
            println!("Assistant: {}", output);
        }
    }

    fn new_session(&mut self) {
        // 调用生成函数
        // ...
        self.sessions.push(Session {
            title: "".to_string(),
            history: Vec::new(),
            kvcache: self.model.new_cache(),
        });

        let session_id = self.sessions.len() - 1;
        self.generator(session_id);
    }

    fn history_session(&mut self) {
        loop {
            println!("Choose a session to view:");
            for (i, session) in self.sessions.iter().enumerate() {
                println!("{}. Session :{}", i + 1, session.title);
            }
            println!("Type 'exit' to return to the previous menu");
            print!("Enter your choice: ");
            io::stdout().flush().unwrap();

            let mut choice = String::new();
            io::stdin().read_line(&mut choice).unwrap();
            let choice = choice.trim();

            if choice.to_lowercase() == "exit" {
                println!("Returning to the main menu...");
                // 返回主菜单或其他逻辑
                return;
            }

            let choice: Result<usize, _> = choice.parse();
            match choice {
                Ok(choice) => {
                    if choice > 0 && choice <= self.sessions.len() {
                        let session_id = choice - 1;
                        self.generator(session_id);
                    } else {
                        println!(
                            "Invalid choice. Please select a valid session number or type 'exit'"
                        );
                    }
                }
                Err(_) => {
                    println!("Invalid input. Please enter a valid number or type 'exit'");
                }
            }
        }
    }
    fn roll_back(&mut self, session_id: usize) {
        // 回滚到上一次的对话
        // ...
        let session = self.sessions.get_mut(session_id).unwrap();
        if session.history.is_empty() {
            println!("No more history to rollback to");
        } else {
            let (kv_count, _) = session.history.pop().unwrap();
            session.kvcache.rollback(kv_count);
        }
        for (_, conversation) in &session.history {
            println!("User: {}", conversation.input);
            println!("Assistant: {}", conversation.output);
        }
    }
}
fn format_prompt(user_input: String) -> String {
    // 实现聊天专用的prompt格式
    let mut prompt = String::from("<|im_start|>user\n");
    prompt.push_str(&user_input);
    prompt.push_str(&String::from("<|im_end|>\n<|im_start|>assistant\n"));
    prompt
}
impl ChatEngine {
    pub fn new() -> Self {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");

        // 先创建临时model实例
        let model = Llama::<f32>::from_safetensors(&model_dir);

        // 现在可以安全初始化结构体字段
        ChatEngine {
            model,
            tokenizer: Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap(),
            sessions: DashMap::new(),
            system_prompt: "You are a chatbot".to_string(),
        }
    }
    pub fn generate(&self, user_id: String, user_input: String) -> String {
        let input = format_prompt(user_input);

        // Use entry API to get or insert the session, which returns a RefMut
        let mut session = self.sessions.entry(user_id.clone()).or_insert_with(|| {
            Session {
                title: user_id.clone(),
                history: Vec::new(),
                kvcache: self.model.new_cache(),
                // Initialize other fields...
            }
        });

        let binding = self.tokenizer.encode(input.clone(), true).unwrap();
        let input_ids = binding.get_ids();

        // Now session is a mutable RefMut, allowing mutable access to kvcache
        let output_ids = self.model.chat_generator(
            input_ids,
            300,
            0.8,
            20,
            1.,
            &mut session.kvcache, // Correctly borrows mutable reference
        );
        self.tokenizer.decode(&output_ids, true).unwrap()
    }
}
