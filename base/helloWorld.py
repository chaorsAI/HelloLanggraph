from typing_extensions import TypedDict, Annotated
from operator import add  # 用于列表追加的reducer

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage


# 1. 定义State的Scheme（结构契约）
class ChatState(TypedDict):
    count: int  # 计数器字段
    messages: Annotated[list, add]  # Message列表，用add做追加reducer

# 2. 定义节点：修改State的函数
def increment_count(state: ChatState) -> ChatState:
    return {"count": state["count"] + 1}  # 直接返回新值（默认覆盖）

def add_human_msg(state: ChatState) -> ChatState:
    msg = HumanMessage(content="你好，LangGraph！")
    return {"messages": [msg]}  # 因reducer是add，实际追加到原列表

# 3. 构建图并运行
builder = StateGraph(ChatState)
builder.add_node("inc", increment_count)
builder.add_node("add_msg", add_human_msg)
builder.add_edge(START, "inc")
builder.add_edge("inc", "add_msg")
builder.add_edge("add_msg", END)

graph = builder.compile()

result = graph.invoke({"count": 0, "messages": []})

print(result)  # 输出: {'count': 1, 'messages': [HumanMessage(content='你好，LangGraph！')]}