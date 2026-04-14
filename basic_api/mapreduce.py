import operator
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from utils.logger import get_logger
from  models import get_lc_o_ali_model_client

logger = get_logger()

# ---------------------- 1. 加载环境 & 模型初始化 ----------------------
load_dotenv()
# 初始化大模型（用于结构化输出）
model = get_lc_o_ali_model_client()

# ---------------------- 2. 提示词模板（统一管理） ----------------------
# 生成1-3个相关主题
SUBJECTS_PROMPT = "生成一个逗号分隔的列表，其中包含1到3个与 {topic} 相关的示例。"
# 根据主题生成笑话
JOKE_PROMPT = "生成一个与 {subject} 相关的消化"
# 从多个笑话中选出最佳
BEST_JOKE_PROMPT = """
下面是一堆关于{topic}的笑话。选择最好的一个！返回最佳ID：{笑话}
"""

# ---------------------- 3. 结构化输出格式（Pydantic） ----------------------
class Subjects(BaseModel):
    """用于接收模型输出：主题列表"""
    subjects: list[str]

class Joke(BaseModel):
    """用于接收模型输出：单条笑话"""
    joke: str

class BestJoke(BaseModel):
    """用于接收模型输出：最佳笑话的索引"""
    id: int = Field(description="Index of the best joke, starting with 0")

# ---------------------- 4. 图状态定义（LangGraph 核心） ----------------------
class OverallState(TypedDict):
    """全局状态：贯穿整个工作流"""
    topic: str                  # 输入的大主题（如 animals）
    subjects: list              # 生成的子主题列表
    jokes: Annotated[list, operator.add]  # 自动合并多条并行生成的笑话
    best_selected_joke: str     # 最终选中的最佳笑话

class JokeState(TypedDict):
    """子任务状态：仅给 generate_joke 使用"""
    subject: str

# ---------------------- 5. 图节点函数（业务逻辑） ----------------------
def generate_topics(state: OverallState):
    """第一步：根据大主题生成 1~3 个子主题"""
    prompt = SUBJECTS_PROMPT.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}

def generate_joke(state: JokeState):
    """第二步：为单个子主题生成笑话（可并行执行）"""
    prompt = JOKE_PROMPT.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}

def best_joke(state: OverallState):
    """第三步：从多条笑话中选出最优的一条"""
    jokes_str = "\n\n".join(state["jokes"])
    prompt = BEST_JOKE_PROMPT.format(topic=state["topic"], jokes=jokes_str)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}

# ---------------------- 6. 条件边：并行分发任务 ----------------------
def continue_to_jokes(state: OverallState):
    """
    核心技术：Map-Reduce 并行执行
    给每个子主题分发一个独立的 generate_joke 任务
    """
    return [Send("generate_joke", {"subject": sub}) for sub in state["subjects"]]

# ---------------------- 7. 构建 LangGraph 工作流 ----------------------
builder = StateGraph(OverallState)

# 添加节点
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)

# 构建流程
builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes)  # 并行分发
builder.add_edge("generate_joke", "best_joke")                      # 汇总结果
builder.add_edge("best_joke", END)

# 编译图
app = builder.compile()

# ---------------------- 8. 运行测试 ----------------------
if __name__ == "__main__":
    # 传入主题，流式输出结果
    for step in app.stream({"topic": "animals"}):
        logger.info(step)