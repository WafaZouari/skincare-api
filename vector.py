from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os 
import pandas as pd
# instead of reading from a CSV, we will use an Exel file
#df = pd.read_csv("products_table_tunisia.csv")
# Read the Excel file
df = pd.read_excel("products_table_tunisia.xlsx")
embeddings= OllamaEmbeddings(model="mxbai-embed-large")

db_location ="./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids= []
    for i, row in df.iterrows():
        document = Document(
        page_content=f"{row['Product']} - {row['Description']} - Solves: {row['Skin Problem']}",metadata={"url": row["Link"]},id=str(i))
        ids.append(str(i))
        documents.append(document)
vector_store = Chroma(
    collection_name="skincare_products",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    vector_store.add_documents(documents =documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)   

