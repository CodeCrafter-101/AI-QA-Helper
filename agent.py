from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import os
from tools import tools

load_dotenv()

# Base LLM
llm = ChatGroq(model ='openai/gpt-oss-20b',api_key=os.getenv("GROQ_API_KEY"))


# LLM + Tools
llm_with_tools = llm.bind_tools(tools)


# 1. Classification Agent
def interface_agent(question: str):
    sys_prompt = """
    You are an AI Question Classifier.
    Classify the question into one category:
    - AI/ML
    - Programming
    - General Knowledge
    - Needs Internet Search

    If the question requires recent information or yesterday's news,
    classify it as "Needs Internet Search".
    """

    response = llm.invoke([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ])

    return response.content


# agent.py 
def answer_agent(question: str):
    system_prompt = """
    You are an AI question answering assistant.
    Give detailed, clear, helpful explanations.
    If the user asks a definition, explain properly with examples.
    If the question needs internet data, you may use Tavily search.
    """

    response = llm_with_tools.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ])

    raw_answer = response.content


    summary = llm.invoke([
    {"role": "system", "content": "Summarize this answer into 4–6 lines."},
    {"role": "user", "content": raw_answer}
    ])


    return response.content


