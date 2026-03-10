#
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull gemma3:1b
#
# pip install langchain langchain-text-splitters langchain-huggingface
# pip install sentence-transformers transformers torch
# pip install langchain-chroma langchain-ollama
# ollama pull gemma3:1b gemma3:4b
#
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. 載入並切分 Markdown 檔案
file_path = "使用牌照稅法_附表一.md"
with open(file_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# 使用標題進行切分
headers_to_split_on = [("##", "章名"), ("####", "條號")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_content)

# 2. 初始化 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={'normalize_embeddings': True}
)

# 3. 建立或載入 ChromaDB
persist_directory = "./chroma_db_tax"
vector_db = Chroma.from_documents(
    documents=md_header_splits, 
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="tax_law_collection"
)
print(f"已成功載入 {len(md_header_splits)} 個 Chunk 至資料庫。")

# 4. 初始化 Ollama 的 Gemma 3 模型； 這邊要用 gemma3:4b 回答的比較好！
llm = ChatOllama(model="gemma3:4b")

# 5. 定義 Prompt 模板
template = """
你是一位專業的稅務諮詢顧問。請根據以下檢索到的法規與表格資料回答使用者的問題。
如果檢索到的資料無法回答問題，請誠實告知你不知道，不要編造資訊。

[相關法條與表格資料]
{context}

[使用者問題]
{question}

請直接開始回答：
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. 互動迴圈
print("=== Gemma 3 RAG 稅務諮詢系統已啟動 ===")
print("系統提示：輸入 exit 可離開程式")

while True:
    query = input("\n請輸入您的稅務問題: ")
    if query.lower() == 'exit':
        break

    # A. 檢索 (Retriever)
    # 搜尋與問題相關性最高的 2 個片段
    results = vector_db.similarity_search(query, k=2)
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    # B. 生成回應
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": query})
    
    print("\n--- AI 顧問回答 ---")
    print(response.content)
