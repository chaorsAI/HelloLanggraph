import operator
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from utils.logger import get_logger
from base.visualization import create_graph_image

logger = get_logger()

# 状态：带自动累加合并能力的列表
class State(TypedDict):
    aggregate: Annotated[list, operator.add]

# 节点函数
def a(state): return {"aggregate": ["A"]}
def b(state): return {"aggregate": ["B"]}
def c(state): return {"aggregate": ["C"]}
def d(state): return {"aggregate": ["D"]}

# 构建分支图，使用.语法
graph = (
    StateGraph(State)
    .add_node(a)
    .add_node(b)
    .add_node(c)
    .add_node(d)

    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("a", "c")
    .add_edge("b", "d")
    .add_edge("c", "d")
    .add_edge("d", END)

    .compile()
)


create_graph_image(graph, "graph_images", name=os.path.splitext(os.path.basename(__file__))[0])

# 执行
logger.info(graph.invoke({"aggregate": []}))