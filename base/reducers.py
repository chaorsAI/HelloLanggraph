from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from utils.logger import get_logger


# 日志
logger = get_logger()

# 1. 定义状态与Reducers（关键！）
class State(TypedDict):
    messages: Annotated[list[str], lambda old, new: old + new]  # 追加合并对话
    count: Annotated[int, lambda old, new: old + new]          # 求和合并计数

# 2. 定义节点（模拟修改状态）
def add_greeting(state: State) -> dict:
    return {"messages": ["User: Hello!"]}  # 节点1：加问候，计数+1

def add_response(state: State) -> dict:
    return {"messages": ["Bot: Hi there!"], "count": 2}  # 节点2：加回复，计数+2

# 3. 构建图（串联节点，用Reducers合并状态）
graph = StateGraph(State)
graph.add_node("greet", add_greeting)
graph.add_node("respond", add_response)
graph.add_edge(START, "greet")
graph.add_edge("greet", "respond")
graph.add_edge("respond", END)

# 4. 运行验证
app = graph.compile()
result = app.invoke({"messages": ["first: what ?"], "count": 3})
logger.info(result)
# 输出：{"messages": ["first: what ?", "User: Hello!", "Bot: Hi there!"], "count": 5}