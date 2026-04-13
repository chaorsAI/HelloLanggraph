from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt

# 大模型相关
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 你的工具类
from models import get_lc_o_ali_model_client
from utils.logger import get_logger

# ======================
# 初始化配置
# ======================
logger = get_logger()
model = get_lc_o_ali_model_client()  # 加载大模型
checkpointer = MemorySaver()          # 记忆存储（保存中断状态）

# ======================
# 状态定义：整个流程只存一个文本
# ======================
class State(TypedDict):
    some_text: str  # 待处理/处理中的文本
    need_human_review: bool  # 新增：动态控制是否需要人类审核

# ======================
# 节点1：AI自动处理（第一步）
# ======================
def auto_process_node(state: State):
    """
    AI自动处理初始请求
    """
    prompt = f"请处理以下内容，简洁回答：{state['some_text']}"
    response = model.invoke([HumanMessage(content=prompt)])

    need_review = "旅游" in state["some_text"]

    return {
        "some_text": response.content,
        "need_human_review": need_review  # 动态控制是否中断
    }

# ======================
# 节点2：动态条件中断（满足才中断）
# ======================
def conditional_human_review(state: State):
    """
    动态中断：
    只有 need_human_review = True 才会等待人类输入
    否则直接跳过
    """
    # ✅ 核心：条件中断
    if state["need_human_review"]:
        # 中断，等待人类输入
        human_input = interrupt({
            "ai_result": state["some_text"],
            "tip": "请输入修改意见（满足条件才会出现)："
        })
        return {"some_text": human_input}

    # 不满足条件 → 不中断，直接返回原内容
    logger.info("✅ 不满足中断条件，跳过人工审核")
    return state

# ======================
# 节点3：AI最终处理（人类输入后再次调用LLM）
# ======================
def final_ai_process_node(state: State):
    """
    核心新增：
    把人类输入再次丢给AI，生成最终优化结果
    """
    prompt = f"""
    你是专业助手，请根据用户的需求，生成最终结果：
    用户需求：{state['some_text']}
    请直接输出最终答案，简洁清晰。
    """
    final_response = model.invoke([HumanMessage(content=prompt)])
    return {"some_text": final_response.content}

# ======================
# 构建流程图
# ======================
graph_builder = StateGraph(State)

# 添加节点
graph_builder.add_node("auto_process", auto_process_node)
graph_builder.add_node("human_review", conditional_human_review)
graph_builder.add_node("final_process", final_ai_process_node)

# 设置流程链路
graph_builder.set_entry_point("auto_process")
graph_builder.add_edge("auto_process", "human_review")   # 自动 → 人工审核
graph_builder.add_edge("human_review", "final_process")  # 人工 → 最终AI处理

# 编译
graph = graph_builder.compile(
    checkpointer=checkpointer,
    # 执行 human_review 之前强制中断
    # interrupt_before=["human_review"]
)

# ======================
# 运行示例（完美可执行）
# ======================
if __name__ == "__main__":
    thread_id = "workflow_001"
    config = {"configurable": {"thread_id": thread_id}}

    # 测试2：包含“旅游” → 会触发中断
    logger.info(f"========== 正常不会触发中断 ===========")
    initial_input = "北京明天的天气怎么样？"
    logger.info(f"初始请求: {initial_input}")

    #执行：AI处理 → 中断
    result = graph.invoke({"some_text": initial_input}, config=config)
    logger.info(f"AI初步处理结果:\n{result['some_text']}")


    # 测试2：包含“旅游” → 会触发中断
    logger.info(f"========== 包含“旅游” → 会触发中断 ===========")
    # 1. 初始输入
    initial_input = "下周去北京旅游怎么样？"
    logger.info(f"初始请求: {initial_input}")

    # 2. 执行：AI处理 → 中断
    result = graph.invoke({"some_text": initial_input}, config=config)
    logger.info(f"AI初步处理结果:\n{result['some_text']}")

    # 3. 人类输入
    human_input = input("请输入你的意见/修改内容：")

    # 4. 恢复流程 → 进入最终AI处理
    resume_result = graph.invoke(Command(resume=human_input), config=config)

    # 5. 输出最终结果
    logger.info(f"\n最终AI优化结果:\n{resume_result['some_text']}")