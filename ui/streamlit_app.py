"""Streamlit 前端界面"""
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG 智能文档问答系统", layout="wide")
st.title("RAG 智能文档问答系统")
st.markdown("上传文档，然后基于文档内容进行智能问答")

with st.sidebar:
    st.header("文档管理")
    uploaded_file = st.file_uploader("选择文档 (PDF/TXT)", type=["pdf", "txt"])
    if uploaded_file and st.button("上传并构建知识库", type="primary"):
        with st.spinner("正在处理文档..."):
            try:
                resp = requests.post(f"{API_URL}/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue())}, timeout=60)
                data = resp.json()
                st.success(data["message"]) if data["status"] == "ok" else st.error(f"上传失败: {data['message']}")
            except Exception as e:
                st.error(f"连接失败: {e}")
    if st.button("清空知识库"):
        try:
            requests.post(f"{API_URL}/clear", timeout=10)
            st.success("知识库已清空")
        except Exception as e:
            st.error(f"操作失败: {e}")
    st.divider()
    st.markdown("### 使用说明\n1. 上传 PDF 或 TXT 文档\n2. 系统自动构建知识库\n3. 在聊天框提问\n4. AI 基于文档内容回答")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                resp = requests.post(f"{API_URL}/query", json={"question": prompt, "top_k": 3}, timeout=30)
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("source_documents"):
                    with st.expander("参考来源"):
                        for src in data["source_documents"]:
                            st.text(src)
                st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
            except Exception as e:
                st.markdown(f"查询失败: {e}")
