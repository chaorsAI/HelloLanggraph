from typing import TypedDict, Literal
import os

from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from visualization import create_graph_image
from utils.logger import get_logger

# ======================
# 1. 定义全局状态（流程唯一数据源）
# ======================
logger = get_logger()

class AgentState(TypedDict):
    question: str                          # 用户问题
    category: Literal["tech", "non-tech"] | None  # 分类：技术/非技术
    response: str | None                   # 最终回答
    need_interrupt: bool | None            # 是否需要中断（用于 resume 演示）

# ======================
# 2. 节点1：智能分类（演示 Command(goto+update)）
# ======================
def classify_node(state: AgentState) -> Command:
    """
    核心演示：
    Command(goto=节点名) → 动态跳转
    Command(update=状态)  → 合并更新状态
    """
    question = state["question"]

    # 规则：包含代码、bug → 技术问题
    if "代码" in question or "bug" in question:
        return Command(
            goto="handle_tech",        # 跳去技术节点
            update={
                "category": "tech",    # 更新分类
                "need_interrupt": True # 需要中断（演示 resume）
            }
        )
    # 其他 → 非技术
    else:
        return Command(
            goto="handle_non_tech",
            update={
                "category": "non-tech",
                "need_interrupt": False
            }
        )

# ======================
# 3. 节点2：处理技术问题（演示 interrupt + resume）
# ======================
def handle_tech(state: AgentState):
    """
    核心演示：
    interrupt() → 暂停流程
    Command(resume=xxx) → 恢复流程
    """
    # 如果需要中断，就暂停，等待人类输入
    if state["need_interrupt"]:
        # 中断！返回给前端/用户
        human_input = interrupt({
            "tip": "请输入人工修改意见",
            "ai_answer": f"已识别技术问题：{state['question']}"
        })

        # 恢复后，用人类输入更新结果
        return {
            "response": f"[技术+人工] {human_input}"
        }

    # 不需要中断 → 直接返回
    return {
        "response": f"[技术] 已处理：{state['question']}"
    }

# ======================
# 4. 节点3：处理非技术问题
# ======================
def handle_non_tech(state: AgentState):
    return {
        "response": f"[非技术] 已转人工：{state['question']}"
    }

# ======================
# 5. 构建工作流
# ======================
builder = StateGraph(AgentState)

builder.add_node("classify", classify_node)
builder.add_node("handle_tech", handle_tech)
builder.add_node("handle_non_tech", handle_non_tech)

builder.set_entry_point("classify")
builder.add_edge("handle_tech", "__end__")
builder.add_edge("handle_non_tech", "__end__")

# 编译（必须加 checkpointer 才能保存中断状态）
app = builder.compile(checkpointer=MemorySaver())

create_graph_image(app, "graph_images", name=os.path.splitext(os.path.basename(__file__))[0])

# ======================
# 6. 运行演示（包含 resume！）
# ======================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-123"}}

    logger.info("开始执行流程...")
    # "Python 代码报错 KeyError"     # 会触发中断
    # "最近的用户增长有点缓慢"
    question = input("请输入现在的问题：")
    # 1. 启动流程
    result = app.invoke(
        {"question": question},
        config=config
    )

    logger.info(f"🔹 首次执行结果：{result}")

    # 检测到中断，才会有恢复
    if result.get("__interrupt__"):
        logger.info("检测到技术问题，转人工")

        # 2. 用户输入 → 恢复流程（核心：resume）
        human_feedback = input("人工9527号修复意见：")
        resume_result = app.invoke(
            Command(resume=human_feedback),  # ✅ resume 恢复中断
            config=config
        )

        # 3. 恢复结果
        logger.info(f"🔹 恢复后最终结果：{resume_result['response']}")

    else:
        logger.info(f"🔹 最终结果：{result['response']}")

