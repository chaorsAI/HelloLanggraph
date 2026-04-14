from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from utils.logger import get_logger


logger = get_logger()

# ---------------------- 1. 定义状态（共享键为"text"） ----------------------
class SharedParentState(TypedDict):
    text: str  # 父图与子图共享的键
    result: str  # 父图私有键（可选）


class SharedSubgraphState(TypedDict):
    text: str  # 与父图共享的键（必须同名）
    processed: str  # 子图私有键（仅子图可见）


# ---------------------- 2. 构建子图（处理共享键"text"） ----------------------
def subgraph_node(state: SharedSubgraphState) -> SharedSubgraphState:
    """子图节点：将共享键"text"转大写，存到私有键"processed"""
    return {
        "text": state["text"].upper(),  # 修改共享键（自动同步到父图）
        "processed": f"Processed: {state['text'].upper()}"  # 子图私有数据
    }


# 编译子图（输入/输出均为SharedSubgraphState）
shared_subgraph = StateGraph(SharedSubgraphState)
shared_subgraph.add_node("process", subgraph_node)
shared_subgraph.add_edge(START, "process")
shared_subgraph.add_edge("process", END)
compiled_shared_subgraph = shared_subgraph.compile()


# ---------------------- 3. 构建父图（集成子图作为节点） ----------------------
def parent_node(state: SharedParentState) -> SharedParentState:
    """父图节点：直接调用子图（共享键自动传递）"""
    return state  # 子图会修改"text"键，父图后续可直接用


parent_builder = StateGraph(SharedParentState)
parent_builder.add_node("call_subgraph", compiled_shared_subgraph)  # 子图作为节点
parent_builder.add_edge(START, "call_subgraph")
parent_builder.add_edge("call_subgraph", END)
compiled_parent = parent_builder.compile()

# ---------------------- 4. 运行验证 ----------------------
if __name__ == "__main__":
    # 初始状态：父图"text"为"hello world"
    initial_state = {"text": "hello world", "result": ""}
    final_state = compiled_parent.invoke(initial_state)

    logger.info("共享状态模式结果：")
    logger.info(f"父图共享键text: {final_state['text']}")  # 输出"HELLO WORLD"（子图修改后同步）
    # 子图私有键"processed"对父图不可见（体现封装性）