"""
OAgent Browser - Web Agent 浏览器工具

一个基于 Playwright 的 Web Agent 浏览器工具，提供：
1. LLM 对话界面用于输入任务
2. 模型调用接口
3. 浏览器动作执行
4. 轨迹记录

使用方法:
    cd oagent_browser
    pip install -r requirements.txt
    playwright install chromium
    python main.py
"""

from .main import OAgentBrowser
from .action_executor import execute_browser_action, parse_tool_call
from .model_interface import call_model, ModelInput, ModelOutput

__version__ = "1.0.0"
__all__ = [
    "OAgentBrowser",
    "execute_browser_action", 
    "parse_tool_call",
    "call_model",
    "ModelInput",
    "ModelOutput"
]
