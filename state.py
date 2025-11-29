from typing import Annotated, TypedDict, List
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
