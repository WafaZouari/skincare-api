from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from vector import retriever
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# CORS config
origins = [
    "http://localhost:3000",
    "https://skincare-front.onrender.com",
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

# ✅ Using HuggingFace Inference API (Mistral model)
model = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-rw-1b",  # small deployable model
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)


# Prompt
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
