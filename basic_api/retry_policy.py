import requests
from typing import Any, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy  # 关键引入

from utils.logger import get_logger
from base.visualization import create_graph_image

logger = get_logger()

# 1. 定义状态和业务函数
class AgentState(TypedDict):
    query: str
    api_response: str

def call_external_api(state: AgentState) -> Dict[str, Any]:
    """一个可能失败的外部API调用函数（纯净的业务逻辑）"""
    response = requests.post(
        "https://api.example.com/query",
        json={"question": state["query"]},
        timeout=10
    )
    response.raise_for_status()  # 如果状态码不是2xx，抛出HTTPError
    return {"api_response": response.json()["answer"]}

# 2. 定义针对此节点的“重试作战计划”
# 方法1：系统内置函数
api_retry_policy = RetryPolicy(
    max_attempts=3,           # 最多尝试3次（首次+2次重试）
    initial_interval=1.0,     # 第一次重试等1秒
    backoff_factor=2.0,       # 指数退避：第二次等2秒，第三次等4秒
    jitter=True,              # 增加随机抖动，避免重试风暴
    retry_on=(requests.exceptions.Timeout,
              requests.exceptions.ConnectionError,
              requests.exceptions.HTTPError)  # 只针对网络和5xx错误重试
)

# 方法2：自定义函数匹配 - 更精细地控制，例如只重试特定HTTP状态码
def retry_on_policy(exception: Exception) -> bool:
    """
    自定义重试判断函数：
    1. 对所有网络连接错误(ConnectionError)和超时(Timeout)进行重试。
    2. 仅对特定HTTP状态码(429, 503)进行重试。
    """
    # 规则1: 如果是连接错误或超时，立即决定重试
    if isinstance(exception, (requests.exceptions.ConnectionError,
                              requests.exceptions.Timeout)):
        return True

    # 规则2: 如果是HTTP错误，则检查状态码
    if isinstance(exception, requests.exceptions.HTTPError):
        # 注意：exception.response 可能为 None（例如在请求未发出时）
        if exception.response is not None:
            if exception.response.status_code in (429, 503):
                return True
    # 其他所有情况，不重试
    return False

# 3. 构建图，并将重试策略绑定到特定节点
builder = StateGraph(AgentState)
# `retry` 参数将策略与节点绑定
builder.add_node("call_api", call_external_api, retry=api_retry_policy)
# builder.add_node("call_api", call_external_api, retry=retry_on_policy)
builder.set_entry_point("call_api")
builder.add_edge("call_api", END)
graph = builder.compile()

# 4. 执行。当`call_api`节点发生超时或连接错误，LangGraph会自动按策略重试。
initial_state = {"query": "LangGraph是什么？", "api_response": ""}
final_state = graph.invoke(initial_state)
print(final_state["api_response"])