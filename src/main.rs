use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use warp::{Filter, Rejection, Reply};

mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

const MAX_HISTORY: usize = 50;

struct ChatSession {
    history: Vec<String>,
    current_kvcache: Mutex<kvcache::KVCache<f32>>,
    kvcache_snapshots: Mutex<Vec<kvcache::KVCache<f32>>>,
}

type HistoryManager = Arc<Mutex<Vec<ChatSession>>>;

#[derive(Debug, Deserialize)]
struct ChatRequest {
    input: String,
    index: i32,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    output: String,
    index: i32,
}

#[derive(Debug, Serialize)]
struct DeleteResponse {
    success: bool,
    message: String,
}

#[derive(Debug, Deserialize)]
struct DeleteRequest {
    index: i32,
}

#[derive(Debug, Serialize)]
struct NewChatResponse {
    index: i32,
}

#[derive(Debug, Deserialize)]
struct RollbackRequest {
    index: i32,
    step: usize,
}

#[derive(Debug, Serialize)]
struct RollbackResponse {
    success: bool,
    message: String,
}

// 构建上下文
fn build_context(history: &[String]) -> String {
    history.join("\n") + "\n<|im_start|>assistant\n"
}

//进行对话交互
async fn handle_chat(
    req: ChatRequest,
    llama: Arc<model::Llama<f32>>,
    tokenizer: Arc<Tokenizer>,
    history_manager: HistoryManager,
) -> Result<impl Reply, Rejection> {
    let system_prompt =
        "<|im_start|>system\nYou are a highly knowledgeable and friendly assistant.";

    let mut manager = history_manager.lock().unwrap();

    // 校验会话索引
    let index = req.index as usize;
    if index >= manager.len() {
        return Err(warp::reject::not_found());
    }

    let session = &mut manager[index];

    // 初始化新会话历史
    if session.history.is_empty() {
        session.history.push(format!("{}<|im_end|>", system_prompt));
    }

    if session.history.len() > MAX_HISTORY {
        session.history = session
            .history
            .split_off(session.history.len() - MAX_HISTORY);
    }

    // 添加用户输入
    session
        .history
        .push(format!("<|im_start|>user\n{}<|im_end|>", req.input));

    // 构建上下文
    let content = build_context(&session.history);

    // 生成回复
    let encoding = tokenizer.encode(&*content, true).unwrap();
    let input_ids = encoding.get_ids();

    // 获取kvcache
    let mut cache_lock = session.current_kvcache.lock().unwrap();

    let output = llama.generate(input_ids, 300, 0.8, 30, 1.0, &mut cache_lock);

    let decoded_output = tokenizer.decode(&output, true).unwrap();

    // 添加历史
    let mut snapshots = session.kvcache_snapshots.lock().unwrap();
    snapshots.push(cache_lock.clone()); // 假设KVCache实现了Clone trait
    session
        .history
        .push(format!("{}<|im_end|>", decoded_output));

    Ok(warp::reply::json(&ChatResponse {
        output: decoded_output,
        index: req.index,
    }))
}

// 创建新会话
async fn new_chat(
    llama: Arc<model::Llama<f32>>,
    history_manager: HistoryManager,
) -> Result<impl Reply, Rejection> {
    let mut manager = history_manager.lock().unwrap();

    let initial_cache = llama.new_cache();
    let snapshots = vec![initial_cache.clone()];

    let new_session = ChatSession {
        history: Vec::new(),
        current_kvcache: Mutex::new(initial_cache),
        kvcache_snapshots: Mutex::new(snapshots),
    };

    let index = manager.len() as i32;
    manager.push(new_session);

    Ok(warp::reply::json(&NewChatResponse { index }))
}

// 删除会话
async fn delete_chat(
    req: DeleteRequest,
    history_manager: HistoryManager,
) -> Result<impl Reply, Rejection> {
    let mut manager = history_manager.lock().unwrap();
    let index = req.index;

    // 边界检查
    if index < 0 || index as usize >= manager.len() {
        return Ok(warp::reply::json(&DeleteResponse {
            success: false,
            message: format!("Invalid index: {}", index),
        }));
    }

    manager.remove(index as usize);

    Ok(warp::reply::json(&DeleteResponse {
        success: true,
        message: format!("Session {} deleted", index),
    }))
}

// 回滚对话
async fn handle_rollback(
    req: RollbackRequest,
    history_manager: HistoryManager,
) -> Result<impl Reply, Rejection> {
    let mut manager = history_manager.lock().unwrap();

    // 校验会话索引
    if req.index < 0 || req.index as usize >= manager.len() {
        return Ok(warp::reply::json(&RollbackResponse {
            success: false,
            message: format!("Invalid index: {}", req.index),
        }));
    }

    let session = &mut manager[req.index as usize];
    let mut snapshots = session.kvcache_snapshots.lock().unwrap();

    // 校验步骤有效性
    if req.step >= snapshots.len() {
        return Ok(warp::reply::json(&RollbackResponse {
            success: false,
            message: format!("Invalid step: {}", req.step),
        }));
    }

    // 恢复缓存和快照
    let mut current_cache = session.current_kvcache.lock().unwrap();
    *current_cache = snapshots[req.step].clone();

    // 截断历史和快照
    session.history.truncate(req.step + 1);
    snapshots.truncate(req.step + 1);

    Ok(warp::reply::json(&RollbackResponse {
        success: true,
        message: format!("Rollback to step {}", req.step),
    }))
}

#[tokio::main]
async fn main() {
    // 模型初始化
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models/chat");
    let llama = Arc::new(model::Llama::<f32>::from_safetensors(&model_dir));
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());

    // 初始化对话
    let history_manager: HistoryManager = Arc::new(Mutex::new({
        vec![ChatSession {
            history: Vec::new(),
            current_kvcache: Mutex::new(llama.new_cache()),
            kvcache_snapshots: Mutex::new(Vec::new()),
        }]
    }));

    // 路由配置
    let chat_route = warp::path!("chat")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_llama(llama.clone()))
        .and(with_tokenizer(tokenizer.clone()))
        .and(with_history(history_manager.clone()))
        .and_then(handle_chat);

    let newchat_route = warp::path!("newchat")
        .and(warp::post())
        .and(with_llama(llama.clone()))
        .and(with_history(history_manager.clone()))
        .and_then(new_chat);

    let delete_route = warp::path!("delete")
        .and(warp::delete())
        .and(warp::body::json())
        .and(with_history(history_manager.clone()))
        .and_then(delete_chat);

    let rollback_route = warp::path!("rollback")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_history(history_manager.clone()))
        .and_then(handle_rollback);

    // CORS配置
    let cors = warp::cors()
        .allow_origin("http://localhost:5173")
        .allow_methods(vec!["POST", "OPTIONS", "DELETE"])
        .allow_headers(vec!["Content-Type"])
        .allow_credentials(true);

    println!("Server started at http://localhost:3030");
    warp::serve(
        chat_route
            .or(newchat_route)
            .or(delete_route)
            .or(rollback_route)
            .with(cors),
    )
    .run(([127, 0, 0, 1], 3030))
    .await;
}

// 辅助函数
fn with_llama(
    llama: Arc<model::Llama<f32>>,
) -> impl Filter<Extract = (Arc<model::Llama<f32>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || llama.clone())
}

fn with_tokenizer(
    tokenizer: Arc<Tokenizer>,
) -> impl Filter<Extract = (Arc<Tokenizer>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || tokenizer.clone())
}

fn with_history(
    history: HistoryManager,
) -> impl Filter<Extract = (HistoryManager,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || history.clone())
}
