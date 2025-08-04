from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import pandas as pd

# ✅ Load Excel file
df = pd.read_excel("products_table_tunisia.xlsx")

# ✅ Use local HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Chroma vector DB setup
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# ✅ Only index documents once
if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=f"{row['Product']} - {row['Description']} - Solves: {row['Skin Problem']}",
            metadata={"url": row["Link"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# ✅ Initialize Chroma store
vector_store = Chroma(
    collection_name="skincare_products",
    embedding_function=embeddings,
    persist_directory=db_location
)

# ✅ Add documents if new
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# ✅ Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
