from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import graph
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="AI QA Helper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(model ='openai/gpt-oss-20b',api_key=os.getenv("GROQ_API_KEY"))

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Welcome to AI QA Helper"}


@app.post("/ask")
def ask_ai(request: Query):
    state = {"messages": [HumanMessage(content=request.question)]}

    result = graph.invoke(state)
    messages = result["messages"]

    if len(messages) < 2:
        return {"error": "The graph did not return enough messages."}


    classification = messages[-2].content
    final_answer = messages[-1].content

    return {
        "category": classification,
        "answer": final_answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

