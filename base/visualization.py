import random
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

from IPython.display import Image, display

# 将编译后 StateGraph 整个工作流程绘制成图
def create_graph_image(graph: CompiledStateGraph, path: str, name: str):
    """
    保存 LangGraph 流程图为 PNG
    - 自动校验目录是否存在
    - 自动补全路径末尾的斜杠 /
    - 自动拼接 name.png
    - 自动创建不存在的文件夹
    """
    # ======================
    # 1. 确保路径末尾有 / 或 \（自动补全斜杠）
    # ======================
    if not path.endswith(os.sep):
        path = path + os.sep  # os.sep 自动适配 Windows/Linux/Mac

    # ======================
    # 2. 校验目录是否存在，不存在则创建
    # ======================
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # 3. 自动处理重名：不存在直接用，存在则加 (1)(2)
    # ======================
    save_file_path = os.path.join(path, f"{name}.png")

    # ======================
    # 4. 保存图片
    # ======================

    # graph.get_graph().draw_mermaid_png(output_file_path=save_file_path)
    graph.get_graph().draw_png(output_file_path=save_file_path)

    # display(Image(filename=save_file_path))
