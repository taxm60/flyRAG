# GPU: NVIDIA A40(12G)
#
# 需承接 sample1.py
# pip install langchain-ollama
#
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. 載入資料庫
persist_directory = "./chroma_db_tax"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 2. 初始化 Ollama 的 Gemma 3 模型
llm = ChatOllama(model="gemma3:1b")

# 3. 定義 Prompt 模板 (讓 LLM 扮演法律顧問)
template = """
你是一位專業的稅務諮詢顧問。請根據以下提供的《使用牌照稅法》相關法條，用白話文回答使用者的問題。
如果法條內容無法回答問題，請誠實告知你不知道。

[相關法條]
{context}

[使用者問題]
{question}

請直接開始回答：
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. 互動迴圈
print("=== Gemma 3 RAG 稅務諮詢系統啟動 ===")
while True:
    query = input("\n請輸入您的問題 (輸入 exit 離開): ")
    if query.lower() == 'exit': break

    # A. 檢索
    results = vectorstore.similarity_search(query, k=2)
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    # B. 生成 (將 context 和 query 丟給 Gemma 3)
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": query})
    
    print("\n--- AI 顧問回答 ---")
    print(response.content)