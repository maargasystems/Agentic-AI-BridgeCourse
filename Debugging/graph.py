from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGroq(model="openai/gpt-oss-120b")

def make_default_graph():
    graph_workflow=StateGraph(State)

    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    agent = graph_workflow.compile()
    return agent

def make_alternate_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a:int, b:int):
        """ add two numbers """
        return a+b

    tool_node = ToolNode([add])

    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages":[model_with_tools.invoke(state["messages"])]}

    def should_continue(state:State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)
    graph_workflow.add_edge("tools", "agent")

    agent = graph_workflow.compile()
    return agent

agent=make_alternate_graph()

if __name__ == "__main__":
    result = agent.invoke({"messages": [("user", "what is 3 + 5?")]})
    print(result["messages"][-1].content)


