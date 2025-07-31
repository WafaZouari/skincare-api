from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

app = FastAPI()

# Allow frontend to fetch from API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SkincareQuery(BaseModel):
    skin_concern: str
    question: str

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template("""
You are a skincare advisor who only recommends products available with delivery to Tunisia.

Skin concern: {skin_concern}

Here are available products:
Product Name | Target Concern(s) | Price (USD/TND) | Buy Link
{products_table}

Question from user: {question}

Your task:
- Recommend only relevant products available in Tunisia for the user's concern.
- Provide prices (convert to TND if needed).
- Propose a complete routine if useful (morning/evening).
- Highlight the best price-quality ratio products.
- Be clear, concise, and avoid recommending unavailable products.
""")
chain = prompt | model

@app.post("/ask")
async def ask_advice(data: SkincareQuery):
    products = retriever.invoke(data.skin_concern)
    result = chain.invoke({
        "skin_concern": data.skin_concern,
        "products_table": products,
        "question": data.question,
    })
    return {"response": result}
