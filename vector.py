# ✅ vector.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Load saved embedding DB only (no indexing on deploy)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="skincare_products",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
