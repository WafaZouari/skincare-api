from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import os

app = FastAPI()

# Configure CORS - adjust these for production
origins = [
    "http://localhost:3000",
    "https://skincare-front.onrender.com",
    # Add your production frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SkincareQuery(BaseModel):
    skin_concern: str
    question: str

# Initialize Mistral - using environment variables for API key
mistral_api_key = os.getenv("MISTRAL_API_KEY")
model = ChatMistralAI(
    model="mistral-large-latest",
    mistral_api_key=mistral_api_key,
    temperature=0.7
)

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
- Always include product links from the metadata.
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
    return {"response": result.content}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}