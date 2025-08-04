from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Make sure retriever still works

import os

# Set your Groq API key (optional if using .env or Render secrets)
os.environ["GROQ_API_KEY"] = "your-groq-api-key"  # Replace with your key or load from env

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
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

# Pydantic model for incoming POST data
class SkincareQuery(BaseModel):
    skin_concern: str
    question: str

# Use Mistral-7B via Groq
model = ChatGroq(
    model_name="mistral-7b",  # You can also try "mixtral-8x7b" for more power
    temperature=0.7
)

# Prompt setup
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

# Chain the prompt with the model
chain = prompt | model

# API endpoint
@app.post("/ask")
async def ask_advice(data: SkincareQuery):
    # Use retriever to get relevant products
    products = retriever.invoke(data.skin_concern)

    # Call the LLM with structured input
    result = chain.invoke({
        "skin_concern": data.skin_concern,
        "products_table": products,
        "question": data.question,
    })

    return {"response": result}

#deploy on Render