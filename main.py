from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Uses pre-built vector DB
import os

# ✅ Environment variable (Render handles this — do not hardcode)
# Set this in your Render Dashboard > Environment
# GROQ_API_KEY must be present

# ✅ FastAPI setup
app = FastAPI()

# ✅ CORS configuration
origins = [
    "http://localhost:3000",
    "https://skincare-front.onrender.com",  # Your deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request body schema
class SkincareQuery(BaseModel):
    skin_concern: str
    question: str

# ✅ Groq LLM setup — use "mixtral-8x7b" or "mistral-7b"
model = ChatGroq(
    model_name="mistral-7b",  # Fast and capable (free on Groq)
    temperature=0.7
)

# ✅ Prompt template
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

# ✅ LangChain chain
chain = prompt | model

# ✅ API endpoint
@app.post("/ask")
async def ask_advice(data: SkincareQuery):
    # 1. Retrieve relevant products
    products = retriever.invoke(data.skin_concern)

    # 2. Run the prompt through the model
    result = chain.invoke({
        "skin_concern": data.skin_concern,
        "products_table": products,
        "question": data.question,
    })

    # 3. Return response
    return {"response": result}
