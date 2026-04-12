# Multiple Schemas：实现“特定节点（a、b）间传递私有字段（如ab_private），其他节点（c）无法访问”

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义主State Schema（全量状态：含私有字段ab_private）
class MainState(TypedDict):
    input_num: int          # 输入字段（来自Input Schema）
    ab_private: int         # 仅a、b传递的私有字段（不在IO Schema）
    final_result: int       # 输出字段（来自Output Schema）
    c_val: int              # c节点处理的公开字段

# 2. 定义Input Schema（仅暴露input_num）
class InputSchema(TypedDict):
    input_num: int

# 3. 定义Output Schema（仅暴露final_result）
class OutputSchema(TypedDict):
    final_result: int

# 4. 定义节点：a写私有字段，b读私有字段，c处理公开字段
def node_a(state: MainState) -> MainState:
    """
    a节点：计算私有字段ab_private（仅a、b用）
    """
    return {"ab_private": state["input_num"] * 2}  # 例：input_num=5 → ab_private=10

def node_b(state: MainState) -> MainState:
    """
    b节点：读ab_private，生成输出final_result
    """
    return {"final_result": state["ab_private"] + 10}  # 10+10=20

def node_c(state: MainState) -> MainState:
    """
    c节点：仅处理公开字段，不碰ab_private
    """
    return {"c_val": state["input_num"] + 5}  # 用input_num计算，例：5+5=10
    return {"c_val": state["ab_private"] + 5}  # 报错，访问不到

# 5. 构建图（绑定主State+IO Schema）
builder = StateGraph(
    state_schema=MainState,
    input_schema=InputSchema,
    output_schema=OutputSchema
)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_edge(START, "a")   # 流程：START→a→b→c→END
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", END)

# 6. 运行验证
graph = builder.compile()
result = graph.invoke({"input_num": 5})  # 输入符合Input Schema


# 输出结果：
print("全量状态（含私有字段）:", result["state"])
# 输出：{'input_num':5, 'ab_private':10, 'final_result':20, 'c_val':10}
print("外部输出（仅final_result）:", result["output"])
# 输出：{'final_result': 20}（ab_private未暴露）
print("c节点结果（未用ab_private）:", result["state"]["c_val"])
# 输出：10（c仅处理input_num，没碰ab_private）