from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pandas as pd
import os

# ✅ Load Excel file
df = pd.read_excel("products_table_tunisia.xlsx")

# ✅ Preprocess data into LangChain Document format
documents = []
for _, row in df.iterrows():
    text = f"Name: {row.get('name', '')}\nDescription: {row.get('description', '')}\nPrice: {row.get('price', '')}"
    metadata = {"id": str(row.get("id", ""))}
    documents.append(Document(page_content=text, metadata=metadata))

# ✅ Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Create Chroma vector store
vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="chroma_db")

# ✅ Create retriever
retriever = vectordb.as_retriever()
