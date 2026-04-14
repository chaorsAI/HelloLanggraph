from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from utils.logger import get_logger


logger = get_logger()

# ====================== 1. 定义独立状态（无任何共享键） ======================
# 核心特点：父子图状态结构完全独立，无同名键，LangGraph 不会自动同步
# 必须手动做状态映射

# 父图独立状态：仅父图使用，与子图无共享字段
class DifferentParentState(TypedDict):
    user_query: str  # 父图输入：用户问题
    answer: str      # 父图输出：最终答案

# 子图独立状态：仅子图使用，与父图无共享字段
class DifferentSubgraphState(TypedDict):
    task: str    # 子图输入：任务内容
    result: str  # 子图输出：处理结果

# ====================== 2. 构建独立子图 ======================
def different_subgraph_node(state: DifferentSubgraphState) -> DifferentSubgraphState:
    """
    子图处理节点（独立运行）
    功能：接收子图自己的 state，处理后返回子图自己的 state
    与父图状态无任何关联
    """
    # 处理逻辑：给 task 增加子图前缀
    return {"result": f"[Subgraph] Processed: {state['task']}"}

# 编译独立子图
different_subgraph = StateGraph(DifferentSubgraphState)
different_subgraph.add_node("process", different_subgraph_node)
different_subgraph.add_edge(START, "process")
different_subgraph.add_edge("process", END)
compiled_different_subgraph = different_subgraph.compile()

# ====================== 3. 构建父图 + 手动状态映射 ======================
def parent_wrapper_node(state: DifferentParentState) -> DifferentParentState:
    """
    父图包装节点：【核心：手动状态映射】
    因为父子图状态完全独立，必须手动：
    1. 父图状态 → 子图输入
    2. 调用子图
    3. 子图输出 → 父图状态
    """
    # ---------------------- 输入映射 ----------------------
    # 父图键：user_query → 子图键：task
    subgraph_input = {"task": state["user_query"]}

    # ---------------------- 调用独立子图 ----------------------
    # 子图与父图状态隔离，不会互相污染
    subgraph_output = compiled_different_subgraph.invoke(subgraph_input)

    # ---------------------- 输出映射 ----------------------
    # 子图键：result → 父图键：answer
    return {
        "answer": subgraph_output["result"],
        "user_query": state["user_query"]  # 保留父图原始输入
    }

# 编译父图：使用包装节点实现子图调用
parent_builder = StateGraph(DifferentParentState)
parent_builder.add_node("call_subgraph", parent_wrapper_node)
parent_builder.add_edge(START, "call_subgraph")
parent_builder.add_edge("call_subgraph", END)
compiled_parent_diff = parent_builder.compile()

# ====================== 4. 运行验证 ======================
if __name__ == "__main__":
    # 父图初始状态
    initial_state = {
        "user_query": "how to use subgraphs?",
        "answer": ""
    }

    # 执行父图
    final_state = compiled_parent_diff.invoke(initial_state)

    # 输出结果
    logger.info("\n" + "="*50)
    logger.info("          LangGraph 独立状态模式运行结果")
    logger.info("="*50)
    logger.info(f"✅ 父图最终 answer: {final_state['answer']}")
    logger.info("="*50)