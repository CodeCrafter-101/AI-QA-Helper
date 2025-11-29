# from langgraph.graph import StateGraph, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_core.messages import AIMessage, HumanMessage
# from state import State
# from agent import interface_agent, answer_agent, llm_with_tools
# from tools import tools


# def interface_node(state: State):
#     user_msg = state["messages"][-1].content
#     category = interface_agent(user_msg)
#     return {"messages": state["messages"] + [AIMessage(content=category)]}


# def answer_node(state: State):
#     user_msg = state["messages"][-2].content
#     response = answer_agent(user_msg)

#     # Tool return handling
#     if hasattr(response, "content"):
#         final = response.content
#     else:
#         final = str(response)

#     return {"messages": state["messages"] + [AIMessage(content=final)]}


# graph_builder = StateGraph(State)

# graph_builder.add_node("interface", interface_node)
# graph_builder.add_node("answer", answer_node)
# graph_builder.add_node("tool_node", ToolNode(tools))

# graph_builder.add_edge(START, "interface")

# graph_builder.add_conditional_edges(
#     "answer",
#     tools_condition,
#     {
#         "tools": "tool_node",
#         "end": END
#     }
# )

# graph_builder.add_edge("tool_node", END)

# graph = graph_builder.compile()



# graph.py
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from state import State
from agent import interface_agent, answer_agent

def interface_node(state: State):
    # The last message should be the user's HumanMessage
    user_msg = state["messages"][-1].content
    category = interface_agent(user_msg)
    # append the classification as an assistant message
    return {"messages": state["messages"] + [AIMessage(content=category)]}

def answer_node(state: State):
    # The original user message is at index 0 (or -2) — find the latest HumanMessage robustly:
    # Here we find the most recent HumanMessage in messages
    human_msgs = [m for m in state["messages"] if m.__class__.__name__.endswith("HumanMessage")]
    if human_msgs:
        user_msg = human_msgs[-1].content
    else:
        # fallback to first message content
        user_msg = state["messages"][0].content

    response_text = answer_agent(user_msg)

    # ensure a string is appended
    final = response_text if isinstance(response_text, str) else str(response_text)

    return {"messages": state["messages"] + [AIMessage(content=final)]}


graph_builder = StateGraph(State)

graph_builder.add_node("interface", interface_node)
graph_builder.add_node("answer", answer_node)

graph_builder.add_edge(START, "interface")
graph_builder.add_edge("interface", "answer")
graph_builder.add_edge("answer", END)

graph = graph_builder.compile()
