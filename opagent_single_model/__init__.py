"""
OAgent Browser - Web Agent Package

Two model backends are available:

    vllm        Full-precision codefuse-ai/OpAgent (Qwen3-32B) via vLLM.
                Use model_interface_vllm.call_model / ModelInput.

    llama_cpp   INT4-quantized model via llama-server HTTP API.
                Use model_interface_llama_cpp.call_model / ModelInput.

Shared utilities:
    action_executor   Browser action parsing and execution.
    browser_runtime   BrowserSession + TrajectoryRecorder (shared by both backends).
"""

from .main import OAgentBrowser
from .action_executor import execute_browser_action, parse_tool_call
from .browser_runtime import BrowserSession, TrajectoryRecorder

__version__ = "2.0.0"
__all__ = [
    # Agent
    "OAgentBrowser",
    # Browser utilities
    "BrowserSession",
    "TrajectoryRecorder",
    # Action utilities
    "execute_browser_action",
    "parse_tool_call",
]
