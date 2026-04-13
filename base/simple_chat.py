from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage,AnyMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from models import get_lc_o_ali_model_client
from utils.logger import get_logger

# 日志
logger = get_logger()

# ================== 类型定义 ==================
class ChatState(TypedDict):
    # messages 自动合并消息的历史记录
    # Annotated：附加信息
    # Sequence：有序列表
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ================== 模型初始化 ==================
model = get_lc_o_ali_model_client()

def handle_user_input(state: ChatState):
    """
    处理用户输入节点（增强健壮性）
    """
    try:
        user_input = input("\n用户输入（输入'Q'结束）: ").strip()
        if user_input.lower() == "Q":
            logger.info(f"正在结束对话...")
            return END
        # 保留历史记录并追加新消息
        return {"messages": [HumanMessage(content=user_input)]}
    except Exception :
        return END


def generate_ai_response(state: ChatState):
    """
    生成AI响应节点（增加错误处理）
    """
    try:
        # 使用最近6条消息保持上下文连贯性
        recent_history = state["messages"][-6:]
        """做个摘要"""
        response = model.invoke(recent_history)
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"系统暂时无法响应，请稍后再试（错误代码：{str(e)[:30]})"
        return {"messages": [AIMessage(content=error_msg)]}


# ================== 对话图构建 ==================
builder = StateGraph(ChatState)

# 节点注册
# RunnableLambda 可不写，add_node 内部会自动包装成 RunnableLambda
builder.add_node("user_input", handle_user_input)
builder.add_node("ai_response", RunnableLambda(generate_ai_response))

# 流程设计
builder.set_entry_point("user_input")
builder.add_edge("user_input", "ai_response")
builder.add_edge("ai_response", END)  # 改为单轮循环

# 编译对话图
conversation = builder.compile()

# ================== 运行逻辑优化 ==================
if __name__ == "__main__":
    # 初始化带时间戳的系统提示
    system_prompt = f"""
    你是一个专业级中文智能助手！
    """
    state = ChatState(messages=[SystemMessage(content=system_prompt)])
    logger.info("===== 智能对话系统已启动 =====")
    logger.info("输入'退出'可随时结束对话\n")

    while True:
        try:
            # 执行对话流程
            result = conversation.invoke(state)
            if result is None or "messages" not in result:
                break
            # 提取最新交互记录
            new_messages = result["messages"][len(state["messages"]):]
            # 打印AI响应
            for msg in new_messages:
                if isinstance(msg, AIMessage):
                    logger.info(f"\n【AI响应】\n{msg.content}\n")
            # 更新对话状态
            state = result
            # 检查退出条件
            if any(isinstance(m, HumanMessage) and m.content.lower() == "退出" for m in state["messages"]):
                break
        except Exception as e:
            logger.info(f"\n系统异常：{str(e)}")
            break

    # 对话结束处理
    logger.info("===== 对话已结束 =====")
    logger.info("【完整对话记录】")

    for i, msg in enumerate(state["messages"][1:]):  # 跳过系统提示
        prefix = "用户：" if isinstance(msg, HumanMessage) else "AI："
        logger.info(f"{i + 1}. {prefix}{msg.content}")