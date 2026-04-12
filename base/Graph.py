from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int

from langgraph.graph import StateGraph

# 实例化一张图
graph_builder = StateGraph(State)

# 编译图
graph = graph_builder.compile()
