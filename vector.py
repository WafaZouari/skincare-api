from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Load Excel
df = pd.read_excel("products_table_tunisia.xlsx")

# Use updated embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

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

    vector_store = Chroma(
        collection_name="skincare_products",
        embedding_function=embeddings,
        persist_directory=db_location
    )
    vector_store.add_documents(documents=documents, ids=ids)
else:
    vector_store = Chroma(
        collection_name="skincare_products",
        embedding_function=embeddings,
        persist_directory=db_location
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
