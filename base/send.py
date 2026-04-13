from typing import TypedDict, List, Union, Dict, Any, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from utils.logger import get_logger

graph_image_path = "graph_images/"

# 日志
logger = get_logger()

# 2. 定义状态（显式包裹）：存消息和笑话
class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # 对话消息
    jokes: Annotated[list[str], operator.add]  # 生成的笑话
    subject: Annotated[str, operator.add]

# 3. 核心节点：同时演示单播/广播（条件分支）
def demo_send(state: State) -> Union[Dict[str, Any], List[Send]]:
    last_msg = state["messages"][-1] if state["messages"] else ""

    # 🔹 单播写法：触发query_order节点
    if "订单" in last_msg:
        return {"send":
                    Send("query_order", {
                        "messages": state["messages"],
                        "jokes": state["jokes"],
                        "subject": ""})
                }  # 显式传递subject，避免字段缺失

    # 🔹 广播写法：触发generate_joke节点3次（并行）
    elif "笑话" in last_msg:
        sends = []
        for s in ["猫", "狗", "咖啡"]:
            # 复制一份 state，把 subject 写进去！
            new_state = {
                "subject": s  # 必须写在这里！
            }
            sends.append(Send("generate_joke", new_state))

        return {"send": sends}

    # 默认分支：触发通用回答
    else:
        return {"send": Send("general_answer", {"messages": state})}


# 4. 单播目标节点：处理订单查询
def query_order(state: State) -> State:
    msg = state["messages"][0]
    return {
        "messages": [f"查订单结果：{msg}已发货"],
        "jokes": state.get("jokes", []),
        "subject": ""
    }

# 5. 广播目标节点：生成单个笑话
def generate_joke(state: State) -> State:
    subject = state.get("subject", "小毛蛋")
    joke = f"{subject}走进酒吧说“来杯牛奶”，老板问“变乖了？”，它说“医生说我得补钙！”"
    return {
        "messages": state.get("messages", []),
        "jokes": state.get("jokes", []) + [joke],
        "subject": ""
    }


# 6. 默认节点：通用回答
def general_answer(state: State) -> State:
    msg = state["messages"][-1]
    return {
        "messages": [f"您的需求：{msg}已记录"],
        "jokes": state.get("jokes", []),
        "subject": ""
    }


# 7. 构建图：注册节点+设置流控
builder = StateGraph(State)
builder.add_node("demo_send", demo_send)
builder.add_node("query_order", query_order)
builder.add_node("generate_joke", generate_joke)
builder.add_node("general_answer", general_answer)
builder.add_edge(START, "demo_send")  # 入口：演示节点

# 添加条件边，让demo_send的send指令驱动路由
def get_next_nodes(state):
    return state["send"]

# 配置demo_send的条件路由（关键：让Send指令生效）
builder.add_conditional_edges(
    "demo_send",
    get_next_nodes,  # 路由函数：返回要触发的目标节点
    {
        "query_order": "query_order",
        "generate_joke": "generate_joke",
        "general_answer": "general_answer"
    }
)

builder.add_edge("query_order", END)  # 单播出口
builder.add_edge("generate_joke", END)  # 单播出口
builder.add_edge("general_answer", END)  # 默认出口

graph = builder.compile()

# 将生成的图片保存到文件
graph_png = graph.get_graph().draw_mermaid_png(output_file_path=f'{graph_image_path}/send.png')

# 8. 运行示例
if __name__ == "__main__":
    # 示例1：单播（含“订单”）
    res_single = graph.invoke({"messages": ["我的订单123没到"], "jokes": [], "subject": ""})
    logger.info("=== 单播结果 ===")
    logger.info("消息:" + str(res_single["messages"]))  # 输出：["我的订单123没到", "查订单结果：我的订单123没到已发货"]
    logger.info("笑话:" + str(res_single["jokes"]))  # 输出：[]

    # 示例2：广播（含“笑话”）
    res_broadcast = graph.invoke({"messages": ["我想听笑话"], "jokes": [], "subject": ""})
    logger.info("\n=== 广播结果 ===")
    logger.info("消息:" + str(res_broadcast["messages"]))  # 输出：[]
    logger.info("笑话:" + str(res_broadcast["jokes"]))  # 输出：3个关于猫、狗、咖啡的笑话

