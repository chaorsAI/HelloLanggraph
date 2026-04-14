import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

from utils.logger import get_logger


logger = get_logger()

# ====================== 1. 状态定义 ======================
# Annotated[list, operator.add] → LangGraph 核心：列表合并（累加）
# 每次节点返回的 list 会自动拼接，而不是覆盖
class State(TypedDict):
    aggregate: Annotated[list, operator.add]


# ====================== 2. 图节点定义 ======================
def node_a(state: State):
    """节点 A：向 aggregate 追加元素 A"""
    logger.info(f'[节点A] 当前集合: {state["aggregate"]}')
    return {"aggregate": ["A"]}


def node_b(state: State):
    """节点 B：向 aggregate 追加元素 B"""
    logger.info(f'[节点B] 当前集合: {state["aggregate"]}')
    return {"aggregate": ["B"]}


def node_c(state: State):
    """节点 C：向 aggregate 追加元素 C"""
    logger.info(f'[节点C] 当前集合: {state["aggregate"]}')
    return {"aggregate": ["C"]}


def node_d(state: State):
    """节点 D：向 aggregate 追加元素 D"""
    logger.info(f'[节点D] 当前集合: {state["aggregate"]}')
    return {"aggregate": ["D"]}


# ====================== 3. 路由函数 ======================
def route_node(state: State) -> Literal["b", END]:
    """
    条件路由：
    - 集合长度 < 10 → 继续执行 B
    - 否则 → 结束流程
    """
    length = len(state["aggregate"])
    logger.info(f"[路由判断] 集合长度: {length}")

    return "b" if length < 10 else END


# ====================== 4. 构建流程图 ======================
builder = StateGraph(State)

# 添加节点
builder.add_node(node_a)
builder.add_node("b", node_b)
builder.add_node(node_c)
builder.add_node(node_d)

# 构建执行流
builder.add_edge(START, "node_a")  # 启动 → A
builder.add_conditional_edges("node_a", route_node)  # A → 路由判断
builder.add_edge("node_b", "node_c")  # B 并行执行 C
builder.add_edge("node_b", "node_d")  # B 并行执行 D
builder.add_edge(["node_c", "node_d"], "node_a")  # C、D 都完成 → 回到 A

# 编译图
graph = builder.compile()

# ====================== 5. 执行流程图 ======================
if __name__ == "__main__":
    try:
        # recursion_limit：LangGraph 最大执行步数（防止死循环）
        # 这里设 100 大于我们需要的 10，所以会正常结束
        result = graph.invoke(
            input={"aggregate": []},
            config={"recursion_limit": 100}
        )
        logger.info("\n✅ 执行完成！最终结果：", result)

    except GraphRecursionError:
        logger.info("\n❌ 超过最大步数限制，触发递归错误")