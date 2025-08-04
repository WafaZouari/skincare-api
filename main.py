from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from vector import retriever
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

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

# ✅ Use Together.ai or any other OpenAI-compatible API key for Mistral
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY"),  # store this in Railway secret
    model="mistralai/Mistral-7B-Instruct-v0.2",
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
""")

# Chain = prompt → llm → output parser
chain = prompt | llm | StrOutputParser()

@app.post("/ask")
async def ask_advice(data: SkincareQuery):
    products = retriever.invoke(data.skin_concern)
    result = chain.invoke({
        "skin_concern": data.skin_concern,
        "products_table": products,
        "question": data.question,
    })
    return {"response": result}
