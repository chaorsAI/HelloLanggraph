# Multiple Schemas：限制输出字段

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. 定义主State Schema（全量状态：含内部/输入/输出字段）
class MainState(TypedDict):
    input_num: int          # 输入字段（来自Input Schema）
    internal_mid: int       # 内部传递字段（场景1用，不暴露给IO）
    final_result: int       # 输出字段（来自Output Schema）

# 2. 定义Input Schema（图输入：仅含input_num）
class InputSchema(TypedDict):
    input_num: int

# 3. 定义Output Schema（图输出：仅含final_result）
class OutputSchema(TypedDict):
    final_result: int

# 4. 定义节点：内部传递特定信息（场景1）
def node_a(state: MainState) -> MainState:
    # 计算中间值，存到internal_mid（仅两节点间用）
    return {"internal_mid": state["input_num"] * 2}

# 5. 定义节点：用内部信息计算最终结果（场景2）
def node_b(state: MainState) -> MainState:
    # 用internal_mid算最终结果，存到final_result
    return {"final_result": state["internal_mid"] + 10}

# 6. 构建图：绑定主State、Input/Output Schema
builder = StateGraph(
    state_schema=MainState,
    input_schema=InputSchema,
    output_schema=OutputSchema
)

builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

# 7. 运行图：输入仅含input_num，输出仅含final_result
graph = builder.compile()
result = graph.invoke({"input_num": 5})  # 输入符合Input Schema

print("全量状态:", result["state"])       # 内部状态：含internal_mid=10
print("输出结果:", result["output"])      # 输出：{"final_result": 20}（符合Output Schema）