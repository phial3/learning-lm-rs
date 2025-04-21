<template>
  <div class="root">
    <aside id="left-aside" :class="{ hidden: isSidebarHidden }">
      <div class="aside-inner">
        <div class="aside-top">
          <button class="new-chat-btn" @click="newChat">
            <span class="icon-new-chart">
              <svg class="iconpark-icon">
                <use href="#add-new-chat"></use>
              </svg>
            </span>
            New Chat
          </button>
          <span id="sidebar-btn" class="sidebar-btn" @click="toggleSidebar">
            <svg class="iconpark-icon">
              <use href="#hide-sidebar"></use>
            </svg>
          </span>
        </div>
        <ul class="chat-list">
          <!-- 动态生成聊天列表 -->
          <li 
            v-for="(chat, index) in chatList" 
            :key="index"
            :class="{ active: currentChatIndex === index }"
            @click="switchChat(index)"
          >
            <span class="icon-chat">
              <svg class="iconpark-icon">
                <use href="#chat-icon"></use>
              </svg>
            </span>
            <div class="chat-list__name">
              Chat {{ index + 1 }}
            </div>
            <div class="icon-chat-opbox">
              <span @click.stop="deleteChat(index)">
                <svg class="iconpark-icon">
                  <use href="#delete"></use>
                </svg>
              </span>
            </div>
          </li>
        </ul>
      </div>
    </aside>
    <section>
      <div id="chat-box" class="chat" ref="chatBox">
        <div v-for="(msg, index) in currentChat.messages" :key="index" class="chat-item" :class="msg.type">
          <div class="chat-inner-item">
            <div class="chat-img">
              <img :src="msg.avatar" style="width:30px" alt="">
            </div>
            <div class="chat-content">
              <div class="text-content" :class="{ 'cursor': msg.isTyping }">
                {{ msg.content }}
              </div>
            </div>
            <span  @click="copyText(msg.content)">
              <svg class="iconpark-icon">
                <use href="#copy"></use>
              </svg>
            </span>
            <span @click.stop="handleTurn(msg, index)">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-bar-up" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M8 10a.5.5 0 0 0 .5-.5V3.707l2.146 2.147a.5.5 0 0 0 .708-.708l-3-3a.5.5 0 0 0-.708 0l-3 3a.5.5 0 1 0 .708.708L7.5 3.707V9.5a.5.5 0 0 0 .5.5zm-7 2.5a.5.5 0 0 1 .5-.5h13a.5.5 0 0 1 0 1h-13a.5.5 0 0 1-.5-.5z"/>
              </svg>
            </span>
          </div>
        </div>
      </div>
      <div class="chat-input">
        <div class="chat-input-content">
          <div class="chat-input-textarea input-area">
            <textarea 
            id="message-textarea" 
            placeholder="Send a message"
            v-model="messageText"
            :style="{ height: textareaHeight + 'px' }"
            @keydown.enter.prevent="handleEnter"
            @input="handleTextareaInput"
            ></textarea>
            <button id="message-btn" class="chat-send-box" :class="{ active: isBtnActive }" @click="sendMessage">
              <span class="chat-send-message">
                <svg class="iconpark-icon">
                  <use href="#send-message"></use>
                </svg>
              </span>
            </button>
          </div>
        </div>
        <p>
          Simple chat model.
        </p>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, nextTick } from 'vue'
import { useToast } from 'vue-toastification'

const toast = useToast()
const isSidebarHidden = ref(false)
const messageText = ref('')
const textareaHeight = ref(24)
const chatBox = ref(null)
const selectedModel = ref('chat')
const currentChatIndex = ref(0)

// 聊天数据
const chatList = reactive([{
  model: 'chat',
  messages: [{
    type: 'answer',
    content: "Hello, I'm an AI assistant. How can I help you today?",
    avatar: 'https://i-1.rar8.net/2023/2/24/e7a2033b-c04e-418c-a1a8-0c3a109557d1.png',
    isTyping: false
  }]
}])

// 计算属性
const isBtnActive = computed(() => messageText.value.trim().length > 0)
const currentChat = computed(() => chatList[currentChatIndex.value])

// 侧边栏切换
const toggleSidebar = () => {
  isSidebarHidden.value = !isSidebarHidden.value
}

// 文本输入处理
const handleTextareaInput = () => {
  // 自动调整高度
  textareaHeight.value = 24
  let height = event.target.scrollHeight
  textareaHeight.value = Math.min(height, 200)
}

// 发送消息
const sendMessage = async () => {
  try{

    const question = messageText.value.trim()
    const url = 'http://localhost:3030/chat';
    // 添加用户提问
    currentChat.value.messages.push({
      type: 'ask',
      content: question,
      avatar: 'https://img.88icon.com/download/jpg/202005/14c48ef66e255e908bce31775f6e8049_512_512.jpg!con'
    })
    
    // 添加回答占位
    const answerIndex = currentChat.value.messages.push({
      type: 'answer',
      content: '',
      avatar: 'https://i-1.rar8.net/2023/2/24/e7a2033b-c04e-418c-a1a8-0c3a109557d1.png',
      isTyping: true
    }) - 1
    messageText.value = ''

    // 处理回答
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input: question ,index: currentChatIndex.value}), // 请求数据
    });
    
    const data = await response.json()
    
    currentChat.value.messages[answerIndex].content = data.output
    currentChat.value.messages[answerIndex].isTyping = false
  }catch (error) {
    console.error('Error:', error);
    toast.error('发送失败，请重试')
  }
    messageText.value = ''
  }
  
// 历史记录管理

// 新建聊天
const newChat = async() => {
  const url = 'http://localhost:3030/newchat';
  chatList.push({
    model: selectedModel.value,
    messages: [{
      type: 'answer',
      content: "Hello, I'm an AI assistant. How can I help you today?",
      avatar: 'https://i-1.rar8.net/2023/2/24/e7a2033b-c04e-418c-a1a8-0c3a109557d1.png',
      isTyping: false
    }]
  })
  currentChatIndex.value = chatList.length - 1
  const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ index: currentChatIndex.value }),
    });
  messageText.value = ''
}

// 删除聊天
const deleteChat = async (index) => {
  if (!confirm('确定要删除这个对话吗？')) return;
  
  try {
    const response = await fetch('http://localhost:3030/delete', {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ index: index }),
    });

    if (!response.ok) throw new Error('删除失败');
    chatList.splice(index, 1);

    // 调整当前选中索引
    if (currentChatIndex.value === index) {
      if (chatList.length > 0) {
        currentChatIndex.value = Math.max(0, index - 1);
      } else {
        await newChat();
      }
    } else if (currentChatIndex.value > index) {
      currentChatIndex.value -= 1;
    }

    toast.success('删除成功');
  } catch (error) {
    console.error('删除失败:', error);
    toast.error('删除失败，请重试');
  }
};

// 切换对话
const switchChat = (index) => {
  currentChatIndex.value = index
}

// 回滚对话
const handleTurn = async (msg, answerIndex) => {
  try {
    // 找到该answer在messages中的位置
    const stepIndex = currentChat.value.messages.findIndex(m => m === msg);
    const newMessages = currentChat.value.messages.slice(0, stepIndex + 1);
    
    // 发送请求
    const response = await fetch('http://localhost:3030/rollback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        index: currentChatIndex.value,
        step: stepIndex  // 传递当前步骤
      }),
    });

    if (!response.ok) throw new Error('回滚失败');
    
    // 更新前端状态
    currentChat.value.messages = newMessages;
    toast.success('对话已回滚');
  } catch (error) {
    console.error('回滚失败:', error);
    toast.error('回滚失败，请重试');
  }
};

// 回车键处理
const handleEnter = (e) => {
  if (e.shiftKey) {
    messageText.value += '\n'
  } else {
    sendMessage()
  }
}

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (chatBox.value) {
      chatBox.value.scrollTop = chatBox.value.scrollHeight
    }
  })
}

// 自动滚动监视
watch(currentChat.value.messages, scrollToBottom, { deep: true })

// 复制文本效果
const copyText = async (text) => {
  try {
    await navigator.clipboard.writeText(text)
    toast.success('复制成功！')
  } catch (err) {
    console.error('复制失败:', err)
    toast.error('复制失败，请手动复制')
  }
}


</script>
