"""
基础 ChatBot + 工具调用 + 对话记忆
功能：AI 可自主判断是否需要联网搜索，自动调用工具，支持多轮记忆
"""
import os
import json
from typing import TypedDict, Annotated

# 环境配置
from dotenv import load_dotenv

# LangChain 核心
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# LangGraph 流式图 + 记忆 + 消息累加
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# 工具导入
from utils.logger import get_logger
from visualization import create_graph_image
from models import get_lc_o_ali_model_client

# ====================== 初始化配置 ======================
# 加载环境变量
load_dotenv()
logger = get_logger()

# 初始化大模型（阿里云/通义千问）
llm = get_lc_o_ali_model_client()

# 初始化联网搜索工具 Tavily
search_tool = TavilySearch(
    max_results=1,
    include_raw_content=True,
    name="Intermediate Answer",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)
tools = [search_tool]

# 给大模型绑定工具
llm_with_tools = llm.bind_tools(tools)

# 对话记忆（保存多轮对话历史）
memory = MemorySaver()

# ====================== 状态定义 ======================
class State(TypedDict):
    """
    图状态结构
    messages: 自动累加的对话消息列表
    """
    messages: Annotated[list, add_messages]

# ====================== 图节点定义 ======================
def chatbot(state: State):
    """AI 对话节点：接收消息 → 调用 LLM → 返回 AI 回复"""
    ai_response = llm_with_tools.invoke(state["messages"])
    return {"messages": [ai_response]}


class BasicToolNode:
    """工具执行节点：自动解析 AI 的工具调用并执行"""
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        # 获取最后一条 AI 消息
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("输入中未找到消息")
        last_message = messages[-1]

        # 执行所有工具调用
        tool_outputs = []
        for tool_call in last_message.tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])

            tool_outputs.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": tool_outputs}

# ====================== 路由判断 ======================
def route_tools(state: State):
    """
    路由函数：判断 AI 是否需要调用工具
    返回：tools → 执行工具 | END → 结束对话
    """
    # 获取最后一条消息
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"未在状态中找到消息: {state}")

    # 检查是否有工具调用
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# ====================== 构建流程图 ======================
def build_graph():
    """构建 LangGraph 工作流图"""
    builder = StateGraph(State)
    tool_node = BasicToolNode(tools=tools)

    # 添加节点
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", tool_node)

    # 添加边（执行流程）
    builder.add_edge(START, "chatbot")        # 开始 → 对话
    builder.add_conditional_edges(             # 对话 → 工具 / 结束
        "chatbot",
        route_tools,
        {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "chatbot")      # 工具执行完 → 回到对话

    # 编译图（带记忆）
    return builder.compile(checkpointer=memory)


# 初始化流程图
graph = build_graph()

# 保存流程图图片（异常不影响主程序）
try:
    filename = os.path.basename(__file__).split(".")[0]
    create_graph_image(graph=graph, path="./graph_images", name=filename)
except Exception:
    pass

# ====================== 交互逻辑 ======================
def stream_response(user_input: str, config: dict):
    """流式输出 AI 回复"""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    ):
        for value in event.values():
            ai_content = value["messages"][-1].content
            logger.info(f"AI: {ai_content}")


# ====================== 主程序 ======================
if __name__ == "__main__":
    # 对话配置：固定 thread_id 实现记忆功能
    chat_config = {"configurable": {"thread_id": "default_chat"}}

    logger.info("🤖 AI 助手已启动（输入 quit/exit/q 退出）")

    while True:
        try:
            user_input = input("User: ")

            # 退出指令
            if user_input.lower() in ["quit", "exit", "q"]:
                logger.info("👋 再见！")
                break

            # 发送消息并获取回复
            stream_response(user_input, chat_config)

        except KeyboardInterrupt:
            logger.info("\n👋 手动退出程序")
            break
        except Exception as e:
            logger.error(f"运行异常: {e}")