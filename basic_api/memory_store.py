# LangGraph + 长时记忆（向量检索）+ 短时对话记忆 的经典案例

# ==============================
# 1. 环境配置与依赖导入
# 功能：加载密钥 + 导入核心库
# ==============================
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # 加载OPENAI_API_KEY

# 向量存储 + 嵌入模型（长时记忆）
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

# LangGraph 工作流核心
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

# AI 对话模型
from models import get_lc_o_ali_model_client, get_ali_embeddings

from utils.logger import get_logger


logger = get_logger()

# ==============================
# 2. 初始化：长时记忆向量库
# 作用：永久存储用户信息，跨会话不丢失
# ==============================
in_memory_store = InMemoryStore(
    index={
        "embed": get_ali_embeddings(),  # 文本向量化
        "dims": 1536,  # 嵌入模型固定输出维度。各个模型可能不同！！！
    }
)

# 初始化 LLM 模型
model = get_lc_o_ali_model_client()

# ==============================
# 3. 核心节点：带记忆的AI对话
# 两大能力：
#  1. 检索历史记忆 → 给AI参考
#  2. 识别remember指令 → 存储新记忆
# ==============================
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # 获取用户ID → 每个用户独立记忆空间
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # 向量检索：从长时记忆中匹配当前对话相关信息
    user_query = state["messages"][-1].content
    memories = store.search(namespace, query=user_query)
    user_info = "\n".join([data.value["data"] for data in memories])

    # 构建系统提示：注入用户记忆
    system_prompt = f"""
    你是乐于助人的助手。
    用户信息（请记住并使用）：{user_info}
    """

    # 判断：用户提到 remember → 自动存储记忆
    last_msg = user_query.lower()
    if "请记住" in last_msg:
        # 提取用户真实输入的内容
        new_memory = user_query.strip()
        # 存储：命名空间 + 唯一ID + 记忆内容
        store.put(namespace, str(uuid.uuid4()), {"data": new_memory})

    # 调用AI生成回复（系统提示 + 对话历史）
    response = model.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    return {"messages": response}

# ==============================
# 4. 构建 LangGraph 工作流
# 单节点流程：开始 → 执行带记忆的对话
# ==============================
workflow = (
    StateGraph(MessagesState)
    .add_node("dialogue_with_memory", call_model)
    .add_edge(START, "dialogue_with_memory")
)

# ==============================
# 5. 编译工作流
# 双记忆配置：
#  - checkpointer：短时对话记忆（聊天上下文）
#  - store：长时永久记忆（用户信息）
# ==============================
agent = workflow.compile(
    checkpointer=MemorySaver(),
    store=in_memory_store
)

# ==============================
# 6. 测试：记忆存储 + 跨线程读取
# ==============================
if __name__ == "__main__":
    # --------------------------
    # 测试1：存储记忆
    # --------------------------
    config1 = {"configurable": {"thread_id": "1", "user_id": "1"}}
    logger.info("【第一轮对话：存储记忆】")
    for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "你好! 请记住: 我就叫 Monkey Bro"}]},
            config1,
            stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # 查看存储的记忆
    logger.info("\n【已存储的长时记忆】")
    user_id = config1["configurable"]["user_id"]
    for memory in in_memory_store.search(("memories", user_id)):
        logger.info(memory.value)

    # --------------------------
    # 测试2：跨线程读取记忆
    # 换thread_id，依然能读取同一user_id的记忆
    # --------------------------
    config2 = {"configurable": {"thread_id": "3", "user_id": "1"}}
    logger.info("\n【第二轮对话：读取记忆】")
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "我的名字叫啥?"}]},
        config2, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()