#[derive(Clone)]
pub struct Message {
    role: String,
    content: String,
}

#[derive(Clone)]
pub struct ChatSession {
    messages: Vec<Message>,
}

impl ChatSession {
    pub fn new() -> Self {
        ChatSession {
            messages: Vec::new(),
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn format_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
}
