from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub  # or use OpenAI/Mistral etc.
from vector import retriever

app = FastAPI()

# ✅ Enable CORS (replace with your frontend domain if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://skincare-front.onrender.com/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define the request model
class AskRequest(BaseModel):
    question: str

# ✅ Choose an LLM (replace with your provider: OpenAI, Mistral, Groq, etc.)
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # or "mistralai/Mistral-7B-Instruct-v0.1"
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# ✅ Build the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Define route
@app.post("/ask")
async def ask_question(request: AskRequest):
    response = qa_chain.run(request.question)
    return {"response": response}
