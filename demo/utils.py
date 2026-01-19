import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union, Callable, Coroutine
import numpy as np
import numpy.typing as npt
import functools
import json
import asyncio
import os
from PIL import Image
import tldextract

@dataclass
class DetachedPage:
    url: str
    content: str  # html

class StateInfo(TypedDict):
    observation: dict[str, Any]
    info: Dict[str, Any]

class Action(TypedDict):
    action_type: int
    coords: npt.NDArray[np.float32]
    element_role: int
    element_name: str
    text: list[int]
    page_number: int
    url: str
    nth: int
    element_id: str
    direction: str
    key_comb: str
    pw_code: str
    answer: str
    raw_prediction: str  # raw prediction from the model
    press_enter_after: int

def with_timeout_legacy(seconds: float):
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                print(f"函数 '{func.__name__}' 执行超时 ({seconds}秒)。")
                try:
                    current_task = asyncio.current_task()
                    if current_task and not current_task.done():
                        current_task.cancel()
                except Exception:
                    pass
                raise
            except Exception as e:
                print(f"函数 '{func.__name__}' 执行时发生未预料的错误: {e}")
                raise
        return wrapper
    return decorator

def change_mainip2ecsip(start_url, REPLACE_WITH_YOUR_HOST):
    if REPLACE_WITH_YOUR_HOST is not None:
        start_url = start_url.replace("REPLACE_WITH_YOUR_HOST", REPLACE_WITH_YOUR_HOST)
    return start_url

# ASCII_CHARSET and other constants usually in constants.py
ASCII_CHARSET = list("!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
SPECIAL_KEYS = ["Enter", "Tab", "ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "Backspace", "Delete", "Escape", "Home", "End", "PageDown", "PageUp"]
FREQ_UNICODE_CHARSET = []

_key2id: dict[str, int] = {
    key: i
    for i, key in enumerate(
        SPECIAL_KEYS + ASCII_CHARSET + FREQ_UNICODE_CHARSET + ["\n"]
    )
}
_id2key: list[str] = sorted(_key2id, key=_key2id.get)

def extract_answer(solution_str):
    if not solution_str:
        return None
    if "<answer>" in solution_str and "</answer>" in solution_str:
        return solution_str.split("<answer>")[1].split("</answer>")[0]
    return None

def smart_resize(height, width, patch_size=14, min_pixels=224*224, max_pixels=1024*1024):
    # Dummy implementation for dependency resolution
    return height, width

def extract_coords_by_index(solution_str, index=0):
    # Dummy implementation
    return [0, 0]

def extract_tool_call_arguments(raw_prediction):
    # Dummy implementation
    if not raw_prediction: return None
    try:
        if "<tool_call>" in raw_prediction:
            return raw_prediction.split("<tool_call>")[1].split("</tool_call>")[0]
    except:
        pass
    return None

