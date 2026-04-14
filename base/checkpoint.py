from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from typing_extensions import TypedDict
from typing import Annotated
from operator import add

from utils.logger import get_logger


logger = get_logger()

# ====================== 1. 定义状态（核心：Reducer 状态合并规则）======================
class State(TypedDict):
    # 无 reducer：新值直接覆盖旧值
    foo: str
    # 带 add reducer：列表自动追加（不会覆盖）
    bar: Annotated[list[str], add]

# ====================== 2. 定义图节点 ======================
def node_a(state: State):
    """节点A：更新状态"""
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    """节点B：更新状态"""
    return {"foo": "b", "bar": ["b"]}

# ====================== 3. 构建工作流 ======================
# 初始化状态图
workflow = StateGraph(State)

# 添加节点
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)

# 定义执行流程
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# ====================== 4. 编译图（开启记忆存储）======================
# 内存记忆器：保存每一步执行状态，支持回溯
checkpointer = InMemorySaver()
# 编译并绑定记忆
graph = workflow.compile(checkpointer=checkpointer)

# ====================== 5. 执行工作流 ======================
# 会话ID：区分不同对话/执行上下文
config = {"configurable": {"thread_id": "thread-1"}}
# 初始状态
initial_state = {"foo": "", "bar": []}

# 执行图
result = graph.invoke(initial_state, config)

# 输出最终结果
logger.info(f"最终状态: {result}")

# ====================== 6. 获取当前最新状态 ======================
snapshot = graph.get_state(config)
logger.info(f"当前状态: {snapshot.values}")
logger.info(f"下一个节点: {snapshot.next}")  # None 表示执行结束

# ====================== 7. 状态历史回溯（时间旅行）======================
history = list(graph.get_state_history(config))
logger.info(f"\n总共有 {len(history)} 个状态快照")

# 遍历历史状态
for i, snap in enumerate(history):
    logger.info(f"\n--- 第 {i} 次状态快照 ---")
    logger.info(f"状态值: {snap.values}")
    logger.info(f"下一步执行: {snap.next}")
    logger.info(f"创建时间: {snap.created_at}")