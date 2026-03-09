# GPU: NVIDIA A40(12G)
#
# pip install langchain-huggingface langchain-chroma langchain-text-splitters sentence-transformers chromadb
#
import os
import shutil
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 0. 清理環境：若已存在資料庫則刪除，避免寫入衝突
persist_directory = "./chroma_db_tax"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print("已清除舊的資料庫...")

# 1. 讀取並切分 Markdown
file_path = "使用牌照稅法.md" 
with open(file_path, "r", encoding="utf-8") as f:
    md_content = f.read()

headers_to_split_on = [("##", "章名"), ("####", "條號")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_content)
print(f"切分完成，共產生 {len(md_header_splits)} 個 Chunk。")

# 2. 初始化 Embedding 模型 (BGE-M3)
print("正在載入 BGE-M3...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={'normalize_embeddings': True}
)

# 3. 寫入 ChromaDB
print("正在將文字轉為向量並寫入 ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=md_header_splits, 
    embedding=embeddings,
    persist_directory=persist_directory
)
print("資料庫建立成功！")

# 4. 進行測試詢問
query = "如果我有身心障礙手冊，我的車可以免徵使用牌照稅嗎？"
results = vectorstore.similarity_search(query, k=1)

print(f"\n--- 系統檢索到的最相關條文 ---")
print(f"條號: {results[0].metadata.get('條號')}")
print(f"內容: {results[0].page_content[:150]}...")