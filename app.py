import streamlit as st
import requests
import hashlib
import datetime

def generate_hash(input_string: str, hash_algorithm: str = 'sha256') -> str:
    hash_func = hashlib.new(hash_algorithm)
    hash_func.update(input_string.encode('utf-8'))
    return hash_func.hexdigest()

def login_page():
    with st.form(key="login_form"):
        name = st.text_input("输入用户名")
        submitted = st.form_submit_button("登录")
        if name and submitted:
            st.session_state.logged_in = True
            if "user_id" not in st.session_state:
                st.session_state.user_id = generate_hash(name)
            if "sessions" not in st.session_state:
                st.session_state.sessions = {
                    "会话0": {
                        "messages": [],
                        "created": datetime.datetime.now()
                    }
                }
            st.session_state.current_session = "会话0"
            st.rerun()

def chat_page():
    with st.sidebar:
        st.subheader("历史会话")
        cols = st.columns(1)  # 每行显示2个会话
        # print(len(st.session_state.sessions))
        for i, session in enumerate(st.session_state.sessions):
            with cols[0]:
                if st.button(
                    session,
                    use_container_width=True,
                    type="primary" if session == st.session_state.current_session else "secondary"
                ):
                    st.session_state.current_session = session
                    st.rerun()

        if st.button("➕新建会话", use_container_width=True):
            new_session = f"会话{len(st.session_state.sessions)}"
            st.session_state.sessions[new_session] = {
                "messages": [],
                "created": datetime.datetime.now()
            }
            st.session_state.current_session = new_session
            st.rerun()
    # 显示历史记录
    history = st.session_state.sessions[st.session_state.current_session]["messages"]
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("向ChatBot发消息"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        data = {"session_id":generate_hash(st.session_state.user_id+st.session_state.current_session),
                "history":"",
                "system_message":"you are a helpful assistant.",
                "user_message":prompt}
        response = requests.post('http://localhost:8080/chat',json=data)
        with st.chat_message("assistant"):
            st.markdown(response.text)
        
        # 添加消息到历史记录
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response.text})


def main():
    # 判断是否已登录
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        # 如果已登录，显示聊天机器人界面
        chat_page()
    else:
        # 否则显示登录界面
        login_page()

if __name__ == "__main__":
    main()