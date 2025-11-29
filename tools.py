from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

load_dotenv()

# Tavily Search Tool
tavily_tool = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5
)

tools = [tavily_tool]
