# pip install streamlit
# streamlit run sample3.py --server.port 80 --server.address 0.0.0.0
#
# sample3.py as follow

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- 網頁設定 ---
st.set_page_config(page_title="法規諮詢系統", page_icon="⚖️")
st.title("⚖️ 《使用牌照稅法》AI 顧問")

# --- 初始化系統 (只載入一次) ---
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma(persist_directory="./chroma_db_tax", embedding_function=embeddings)
    llm = ChatOllama(model="gemma3:1b")
    return vectorstore, llm

vectorstore, llm = load_system()

# --- 對話紀錄儲存 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 顯示歷史訊息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 使用者輸入區 ---
if prompt := st.chat_input("請輸入關於牌照稅的問題..."):
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 檢索與生成
    with st.chat_message("assistant"):
        results = vectorstore.similarity_search(prompt, k=2)
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        template = "你是專業稅務顧問，根據法條回答：\n{context}\n問題：{question}"
        full_prompt = template.format(context=context_text, question=prompt)
        
        response = llm.invoke(full_prompt)
        st.markdown(response.content)
        
    st.session_state.messages.append({"role": "assistant", "content": response.content})