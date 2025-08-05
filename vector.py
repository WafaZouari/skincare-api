from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os 
import pandas as pd

# Initialize embeddings with Mistral
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

# Read the Excel file
df = pd.read_excel("products_table_tunisia.xlsx")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=f"{row['Product']} - {row['Description']} - Solves: {row['Skin Problem']}",
            metadata={
                "url": row["Link"],
                "product": row["Product"],
                "price": row.get("Price", "N/A"),
                "concerns": row["Skin Problem"]
            }
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="skincare_products",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)