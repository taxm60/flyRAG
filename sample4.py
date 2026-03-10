#
# pip install langchain langchain-text-splitters langchain-huggingface
# pip install sentence-transformers transformers torch
# pip install langchain-chroma
#
# 處理原始資料為表格(table)
#
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 載入與切分 Markdown
file_path = "使用牌照稅法_附表一.md" 
with open(file_path, "r", encoding="utf-8") as f:
    md_content = f.read()

headers_to_split_on = [("##", "章名"), ("####", "條號")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_content)

# 2. 初始化 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={'normalize_embeddings': True}
)

# 3. 寫入 ChromaDB
# 將資料存入 ./chroma_db 目錄，並指定 collection 名稱
vector_db = Chroma.from_documents(
    documents=md_header_splits, 
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="tax_law_collection"
)

print(f"已成功將 {len(md_header_splits)} 個 Chunk 寫入 ChromaDB。")

# 4. 測試檢索 (選用)
query = "小客車稅額是多少？"
docs = vector_db.similarity_search(query, k=2)
print(f"查詢結果: {docs[0].page_content}")
