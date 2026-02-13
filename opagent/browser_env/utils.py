import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image
import asyncio
import functools
from typing import Callable, Any, Coroutine
import json
import tldextract
from .tbase_cache import cookie_cache, origins_cache
try:
    from vertexai.preview.generative_models import Image as VertexImage
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image')
from playwright._impl._api_structures import StorageState
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

def change_mainip2ecsip(start_url, REPLACE_WITH_YOUR_HOST):

    # "start_url": "http://REPLACE_WITH_YOUR_HOST:7780/admin",
    # 替换为 "http://REPLACE_WITH_YOUR_HOST:7780/admin" 中的 "REPLACE_WITH_YOUR_HOST" 为 options["REPLACE_WITH_YOUR_HOST"]
    #首先获取到start_url中的hostname
    if REPLACE_WITH_YOUR_HOST is not None:
        start_url = start_url.replace("REPLACE_WITH_YOUR_HOST", REPLACE_WITH_YOUR_HOST)
    return start_url

def get_storage_state_from_config_file(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return get_storage_state_from_start_url(config['start_url'])

def get_storage_state_from_start_url(start_url):
    local_storage_state_path = "/ainative/muti-modal/peiheng/459896/projects/gui_agent/visualwebarena/config_files/wa/storage_state/"
    domain = extract_main_domain(start_url)
    #print(f"INFO [utils]: domain: {domain}")
    if domain is None:
        storage_state: StorageState = {
            "cookies": [],
            "origins": []
        }
        return storage_state
    if os.path.exists(os.path.join(local_storage_state_path, f"{domain}.json")):
        with open(os.path.join(local_storage_state_path, f"{domain}.json"), 'r') as f:
            storage_state = json.load(f)
        return storage_state
    else:
        cookies = cookie_cache.get_session(domain)
        origins = origins_cache.get_session(domain)
        storage_state: StorageState = {
            "cookies": cookies,
            "origins": origins
        }
        #logger.info(f"INFO [utils]: storage_state: ---\n {storage_state} \n---")
        return storage_state

def extract_main_domain(url):
    clean_url = url.strip().replace(" ", "")
    if not clean_url.startswith(('http://', 'https://')):
        clean_url = 'http://' + clean_url
    try:
        extracted = tldextract.extract(clean_url)
        if not extracted.domain:
            return None
        main_domain = f"{extracted.domain}.{extracted.suffix}"
        return main_domain
    except Exception as e:
        return None

@dataclass
class DetachedPage:
    url: str
    content: str  # html


def with_timeout_legacy(seconds: float):
    """
    一个兼容旧版Python的异步装饰器，使用 asyncio.wait_for 提供超时。

    :param seconds: 超时时间（秒）。
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # 使用 asyncio.wait_for 包裹函数调用
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                print(f"函数 '{func.__name__}' 执行超时 ({seconds}秒)。")
                # 对于超时情况，尝试取消可能正在运行的任务
                try:
                    # 获取当前任务
                    current_task = asyncio.current_task()
                    if current_task and not current_task.done():
                        current_task.cancel()
                except Exception:
                    pass
                raise
            except asyncio.CancelledError:
                print(f"函数 '{func.__name__}' 被取消。")
                raise
            except Exception as e:
                print(f"函数 '{func.__name__}' 执行时发生未预料的错误: {e}")
                raise
        return wrapper
    return decorator

@beartype
def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def pil_to_vertex(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None
    center: list[float] | None


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: int
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None
    center: list[float] | None


class BrowserConfig(TypedDict):
    win_upper_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]

Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]



