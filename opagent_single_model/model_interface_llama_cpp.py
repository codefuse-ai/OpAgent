#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""llama.cpp HTTP-backed model interface for OpAgent.

Talks to a running ``llama-server`` instance via its OpenAI-compatible
``/v1/chat/completions`` API.  This is the **INT4-quantized** path:

    Browser screenshot  →  HTTP POST to llama-server  →  parsed action

To use this backend, start ``llama-server`` separately (or let the Gradio
app start it automatically), then set the environment variables below.

Environment variables (all optional — defaults shown):
    OPAGENT_LLAMA_SERVER_URL          http://127.0.0.1:18080
    OPAGENT_LLAMA_COMPLETION_ENDPOINT /v1/chat/completions
    OPAGENT_LLAMA_MODEL_NAME          qwen-vl
    OPAGENT_LLAMA_MODEL_PATH          ~/workspace/OpAgent/OpAgent-32B-Q4/OpAgent.Q4_K_M.gguf
    OPAGENT_LLAMA_MM_PROJ_PATH        (empty → not required)
    OPAGENT_LLAMA_CTX_SIZE            4096
    OPAGENT_LLAMA_TEMPERATURE         0
    OPAGENT_LLAMA_MAX_TOKENS          512
    OPAGENT_LLAMA_TIMEOUT             600
    OPAGENT_REQUIRE_LOCAL_MODEL_FILES auto   (auto|1|0)
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from loguru import logger
import requests
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration (read from env with sensible defaults)
# ---------------------------------------------------------------------------

LLAMA_SERVER_URL: str = os.getenv("OPAGENT_LLAMA_SERVER_URL", "http://127.0.0.1:18080").rstrip("/")
LLAMA_COMPLETION_ENDPOINT: str = os.getenv("OPAGENT_LLAMA_COMPLETION_ENDPOINT", "/v1/chat/completions")
LLAMA_MODEL_NAME: str = os.getenv("OPAGENT_LLAMA_MODEL_NAME", "qwen-vl")
LLAMA_MODEL_PATH: str = os.path.expanduser(
    os.getenv("OPAGENT_LLAMA_MODEL_PATH", "~/workspace/OpAgent/OpAgent-32B-Q4/OpAgent.Q4_K_M.gguf")
)
LLAMA_MM_PROJ_PATH: str = os.path.expanduser(os.getenv("OPAGENT_LLAMA_MM_PROJ_PATH", "")).strip()
LLAMA_CTX_SIZE: int = int(os.getenv("OPAGENT_LLAMA_CTX_SIZE", "4096"))
LLAMA_TEMPERATURE: float = float(os.getenv("OPAGENT_LLAMA_TEMPERATURE", "0"))
LLAMA_MAX_TOKENS: int = int(os.getenv("OPAGENT_LLAMA_MAX_TOKENS", "512"))
LLAMA_TIMEOUT: int = int(os.getenv("OPAGENT_LLAMA_TIMEOUT", "600"))
LLAMA_REQUIRE_LOCAL_FILES: str = os.getenv("OPAGENT_REQUIRE_LOCAL_MODEL_FILES", "auto").strip().lower()

# ---------------------------------------------------------------------------
# Tool list sent in every prompt
# ---------------------------------------------------------------------------

TOOL_LIST = [
    {
        "name": "click",
        "description": "Click on an element with coordinates on the screenshot of the webpage.",
        "parameters": {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "list",
                    "description": "The coordinates of the element in the image to click: [x,y]"
                }
            },
            "required": ["coords"]
        }
    },
    {
        "name": "type",
        "description": "Type content into a field with a specific id.",
        "parameters": {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "list",
                    "description": "The coordinates of the element in the image to click: [x,y]"
                },
                "content": {
                    "type": "string",
                    "description": "Text to be typed"
                },
                "press_enter_after": {
                    "type": "integer",
                    "description": "Whether to press Enter after typing (1 by default, 0 to disable)",
                    "default": 0
                }
            },
            "required": ["coords", "content"]
        }
    },
    {
        "name": "hover",
        "description": "Hover over an element with the coordinates",
        "parameters": {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "list",
                    "description": "The coordinates of the element in the image to hover: [x,y]"
                }
            },
            "required": ["coords"]
        }
    },
    {
        "name": "press",
        "description": "Simulate pressing a key or a key combination",
        "parameters": {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "list",
                    "description": "The coordinates of the element in the image to hover: [x,y]"
                },
                "key": {
                    "type": "string",
                    "description": "a key or a key combination to press (e.g., 'ctrl+v' or 'enter')"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "scroll",
        "description": "Scroll the page up or down",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll"
                },
                "distance": {
                    "type": "integer",
                    "description": "The scroll distance"
                }
            },
            "required": ["direction", "distance"]
        }
    },
    {
        "name": "hscroll",
        "description": "Scroll the page horizontally",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Direction to scroll horizontally"
                },
                "distance": {
                    "type": "integer",
                    "description": "The scroll distance"
                }
            },
            "required": ["direction", "distance"]
        }
    },
    {
        "name": "new_tab",
        "description": "Open a new, empty browser tab",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "tab_focus",
        "description": "Switch browser focus to a specific tab",
        "parameters": {
            "type": "object",
            "properties": {
                "tab_index": {
                    "type": "integer",
                    "description": "Index of the tab to focus"
                }
            },
            "required": ["tab_index"]
        }
    },
    {
        "name": "go_back",
        "description": "Navigate to the previously viewed page",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "go_forward",
        "description": "Navigate to the next page after a 'go_back' action",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "move_to",
        "description": "Move the cursor to a specific location without clicking",
        "parameters": {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "list",
                    "description": "The coordinates to move the cursor to: [x,y]"
                }
            },
            "required": ["coords"]
        }
    },
    {
        "name": "double_click",
        "description": "Perform a double click at a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "list",
                    "description": "The coordinates to double click: [x,y]"
                }
            },
            "required": ["coords"]
        }
    },
    {
        "name": "goto",
        "description": "Navigate to a specific URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "wait",
        "description": "Wait for the change to happen",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "The seconds to wait"
                }
            },
            "required": ["seconds"]
        }
    },
    # answer is a special action required for task completion
    {
        "name": "answer",
        "description": "Task is completed, provide the final answer",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The final answer or result of the task"
                }
            },
            "required": ["content"]
        }
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelInput:
    """All information needed to build a single-step agent prompt."""
    screenshot_base64: str            # PNG screenshot encoded as base64 string
    query: str                        # The user's original task description
    history: List[Dict[str, Any]]     # Recent action history (last few steps)
    current_url: str                  # URL currently shown in the browser


@dataclass
class ModelOutput:
    """Structured output returned after one llama-server call."""
    raw_response: str                       # Content field returned by llama-server
    think_content: Optional[str]            # Content inside <think>…</think>
    tool_call: Optional[Dict[str, Any]]     # Parsed & normalised action dict


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(model_input: ModelInput) -> str:
    """Construct the text prompt sent to llama-server."""
    screenshot_bytes = base64.b64decode(model_input.screenshot_base64)
    with Image.open(BytesIO(screenshot_bytes)) as image:
        original_width, original_height = image.size

    history_blocks: List[str] = []
    for item in model_input.history[-2:]:  # pass last 2 steps only
        history_action = {
            "name": item.get("action_type", ""),
            "arguments": {},
        }
        params = item.get("params") or {}
        x = params.get("x")
        y = params.get("y")
        if x is not None and y is not None:
            history_action["arguments"]["coords"] = [
                int(x / original_width * 1000),
                int(y / original_height * 1000),
            ]
        action_type = item.get("action_type", "")
        if action_type == "type":
            history_action["arguments"]["content"] = params.get("text", params.get("content", ""))
            history_action["arguments"]["press_enter_after"] = 1 if params.get("press_enter", False) else 0
        elif action_type in {"scroll", "hscroll"}:
            history_action["arguments"]["direction"] = params.get("direction", "down")
            history_action["arguments"]["distance"] = params.get("distance", 300)
        elif action_type == "press":
            history_action["arguments"]["key"] = params.get("key", "Enter")
        elif action_type == "wait":
            history_action["arguments"]["seconds"] = params.get("seconds", 2)
        elif action_type == "goto":
            history_action["arguments"]["url"] = params.get("url", "")
        elif action_type == "tab_focus":
            history_action["arguments"]["tab_index"] = params.get("tab_index", 0)
        elif action_type == "answer":
            history_action["arguments"]["content"] = params.get("content", "")

        history_blocks.append(
            f"Step {len(history_blocks) + 1}:\n"
            f"<think>Executing {action_type} action</think>"
            f"<tool_call>{json.dumps(history_action, ensure_ascii=False)}</tool_call>"
        )

    tool_desc = json.dumps(TOOL_LIST, ensure_ascii=False, indent=2)
    parts = [
        "You are an excellent web agent. Based on the web screen shot and content, your need call the single, most appropriate tool for the current step to make progress on the user's request.",
        "Output the thinking process in <think> </think> tags, and for each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:",
        '<think> ... </think><tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>',
        "You are provided with function signatures within <tools></tools> XML tags:",
        f"<tools>\n{tool_desc}\n</tools>",
        "<image>",
        "Please generate the next move according to the UI screenshot, instruction and previous actions.",
        f"Instruction: '{model_input.query}'.",
    ]
    parts.extend(history_blocks if history_blocks else ["None"])
    parts.append(f"Step {len(history_blocks) + 1}:")
    parts.append("NOTES: 1): If you see a search bar, simply use the type tool to input your query. There's no need to click the search bar first.")

    return "\n".join(parts)


def build_chat_payload(model_input: ModelInput) -> Dict[str, Any]:
    """Build the OpenAI-compatible chat/completions payload."""
    return {
        "model": LLAMA_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(model_input)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{model_input.screenshot_base64}"},
                    },
                ],
            }
        ],
        "temperature": LLAMA_TEMPERATURE,
        "max_tokens": LLAMA_MAX_TOKENS,
    }


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

_TERMINAL_ACTIONS = {"answer", "stop", "finish", "done"}


def _convert_model_format(
    model_output: Dict[str, Any],
    original_width: int = 1280,
    original_height: int = 800,
) -> Dict[str, Any]:
    """Convert the raw {name, arguments} tool_call to the internal action dict."""
    name = model_output.get("name", "")
    arguments = model_output.get("arguments", {})
    result: Dict[str, Any] = {"action_type": name}

    if name not in _TERMINAL_ACTIONS:
        coords = arguments.get("coords")
        if coords and isinstance(coords, list) and len(coords) >= 2:
            coord_x = coords[0]
            coord_y = coords[1]
            if 0 <= coord_x <= 1000 and 0 <= coord_y <= 1000:
                result["x"] = round(coord_x / 1000.0 * original_width)
                result["y"] = round(coord_y / 1000.0 * original_height)
            else:
                logger.warning(
                    f"Model returned non-normalized coords {coords}; treating them as pixel coords"
                )
                result["x"] = round(coord_x)
                result["y"] = round(coord_y)

    if name == "type":
        result["text"] = arguments.get("content", "")
        result["press_enter"] = arguments.get("press_enter_after", 0) == 1
    elif name in {"scroll", "hscroll"}:
        result["direction"] = arguments.get("direction", "down")
        result["distance"] = arguments.get("distance", 300)
    elif name == "press":
        result["key"] = arguments.get("key", "Enter")
    elif name == "wait":
        result["seconds"] = arguments.get("seconds", 2)
    elif name == "goto":
        result["url"] = arguments.get("url", "")
    elif name == "tab_focus":
        result["tab_index"] = arguments.get("tab_index", 0)
    elif name in _TERMINAL_ACTIONS:
        result["action_type"] = "answer"
        result["content"] = arguments.get("content", "")
        result.pop("x", None)
        result.pop("y", None)

    return result


def _extract_json_objects(text: str) -> List[str]:
    """Extract balanced {…} blocks from arbitrary text."""
    results: List[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                results.append(text[start: i + 1])
                start = -1
    return results


def parse_model_response(
    raw_response: str,
    original_width: int = 1280,
    original_height: int = 800,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parse think-content and tool_call from raw llama-server output."""
    think_content: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None

    # Strip Qwen chat markers
    clean = re.sub(r"<\|im_start\|>\w+\n?|<\|im_end\|>", "", raw_response).strip()

    # --- Think content ---
    for pattern in [r"<think>(.*?)</think>", r"\[Start thinking\](.*?)\[End thinking\]"]:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            think_content = m.group(1).strip()
            break

    # --- Tool call: priority 1 — explicit <tool_call> wrapper ---
    m = re.search(r"<tool_call>(.*?)</tool_call>", clean, re.DOTALL)
    if m:
        try:
            tool_call = _convert_model_format(json.loads(m.group(1).strip()), original_width, original_height)
        except Exception as exc:
            logger.warning(f"Failed to parse <tool_call> JSON: {exc}")

    # --- Priority 2 — any JSON object with "name" + "arguments" ---
    if tool_call is None:
        for candidate in _extract_json_objects(clean):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and parsed.get("name") and "arguments" in parsed:
                    tool_call = _convert_model_format(parsed, original_width, original_height)
                    break
            except Exception:
                continue

    # --- Priority 3 — single-line JSON fragments ---
    if tool_call is None:
        for fragment in re.findall(r"\{[^\n]+\}", clean):
            try:
                parsed = json.loads(fragment)
                if isinstance(parsed, dict) and parsed.get("name"):
                    tool_call = _convert_model_format(parsed, original_width, original_height)
                    break
            except Exception:
                continue

    return think_content, tool_call


# ---------------------------------------------------------------------------
# Server health check helpers
# ---------------------------------------------------------------------------

def ensure_llama_server_available() -> Dict[str, str]:
    """Verify ``llama-server`` is reachable.

    Local file existence checks are only performed when the server URL points
    to the local machine (or when ``OPAGENT_REQUIRE_LOCAL_MODEL_FILES=1``).
    """
    model_path = Path(LLAMA_MODEL_PATH)

    parsed = urlparse(LLAMA_SERVER_URL)
    hostname = (parsed.hostname or "").strip().lower()
    is_local_server = hostname in {"", "127.0.0.1", "localhost", "0.0.0.0", "::1"}

    if LLAMA_REQUIRE_LOCAL_FILES in {"1", "true", "yes", "on"}:
        require_local_files = True
    elif LLAMA_REQUIRE_LOCAL_FILES in {"0", "false", "no", "off"}:
        require_local_files = False
    else:
        require_local_files = is_local_server

    if require_local_files:
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")
        if LLAMA_MM_PROJ_PATH and not Path(LLAMA_MM_PROJ_PATH).exists():
            raise FileNotFoundError(f"mmproj file not found: {LLAMA_MM_PROJ_PATH}")

    health_urls = [f"{LLAMA_SERVER_URL}/health", f"{LLAMA_SERVER_URL}/"]
    last_error: Optional[Exception] = None
    for health_url in health_urls:
        try:
            response = requests.get(health_url, timeout=5)
            if response.ok:
                return {
                    "llama_server_url": LLAMA_SERVER_URL,
                    "model_path": str(model_path),
                    "require_local_files": str(require_local_files).lower(),
                }
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"llama-server not reachable at {LLAMA_SERVER_URL}: {last_error}")


def ensure_llama_cli_available() -> Dict[str, str]:
    """Compatibility alias."""
    return ensure_llama_server_available()


def set_llama_server_url(url: str) -> str:
    """Update the llama-server URL at runtime."""
    global LLAMA_SERVER_URL
    LLAMA_SERVER_URL = url.rstrip("/")
    os.environ["OPAGENT_LLAMA_SERVER_URL"] = LLAMA_SERVER_URL
    return LLAMA_SERVER_URL


def _completion_url() -> str:
    endpoint = LLAMA_COMPLETION_ENDPOINT
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{LLAMA_SERVER_URL}{endpoint}"


def _call_completion_api(model_input: ModelInput) -> str:
    payload = build_chat_payload(model_input)
    response = requests.post(_completion_url(), json=payload, timeout=LLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"llama-server returned empty content: {data}")
    return content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_model(model_input: ModelInput) -> ModelOutput:
    """Run one agent step: build prompt → call llama-server → parse output.

    This is a *synchronous* function; use ``asyncio.to_thread`` to call it
    from an async context without blocking the event loop.
    """
    ensure_llama_server_available()
    screenshot_bytes = base64.b64decode(model_input.screenshot_base64)
    with Image.open(BytesIO(screenshot_bytes)) as image:
        original_width, original_height = image.size

    logger.info(f"[llama-server] Calling model for task: {model_input.query[:80]!r}")

    raw_response = _call_completion_api(model_input)

    logger.info(f"[llama-server] Raw response ({len(raw_response)} chars): {raw_response[:500]!r}")
    think_content, tool_call = parse_model_response(raw_response, original_width, original_height)

    if tool_call:
        logger.info(f"[llama-server] Parsed tool_call: {tool_call}")
    else:
        logger.warning(f"[llama-server] No tool_call parsed. Full model output: {raw_response!r}")

    return ModelOutput(
        raw_response=raw_response,
        think_content=think_content,
        tool_call=tool_call,
    )
