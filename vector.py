from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
# import pandas as pd  # Uncomment if using preprocessing
import os

# ✅ Embedding model (light enough for deployment)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Chroma vector DB configuration
db_location = "./chroma_langchain_db"
add_documents = False  # ⚠️ Do NOT add documents during deployment

# ✅ Load existing vector store
vector_store = Chroma(
    collection_name="skincare_products",
    embedding_function=embeddings,
    persist_directory=db_location
)

# ✅ Retriever for skin concern queries
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 🔧 OPTIONAL: Run this block locally **once** to create the vector DB
"""
if not os.path.exists(db_location):
    # Load your product data from Excel
    df = pd.read_excel("products_table_tunisia.xlsx")

    # Convert each product to a document
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

    # Create and persist the Chroma DB
    vector_store = Chroma(
        collection_name="skincare_products",
        embedding_function=embeddings,
        persist_directory=db_location
    )
    vector_store.add_documents(documents=documents, ids=ids)
"""
