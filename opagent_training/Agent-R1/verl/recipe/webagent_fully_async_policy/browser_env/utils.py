import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict

import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image, ImageDraw
import asyncio
import functools
from typing import Callable, Any, Coroutine
import json
import tldextract
try:
    from vertexai.preview.generative_models import Image as VertexImage
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image')
from playwright._impl._api_structures import StorageState
import os
import logging
import glob
"""
Utility functions for working with tools
"""
import re
import inspect
from typing import Callable, Dict, Any, Optional, Tuple
import os
import json_repair
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
#logger.setLevel("DEBUG")
try:
    from transformers.utils import get_json_schema
except ImportError:
    raise ImportError(
        "The transformers library is required for this functionality. "
        "Please install it with: pip install transformers>=4.35.0"
    )

from recipe.webagent_fully_async_policy.browser_env.tool_base import Tool
import math

NUM_NODES = os.environ.get("NUM_NODES", 1)
TENSORBOARD_DIR = os.environ.get("TENSORBOARD_DIR", "")
TASK_ID = os.environ.get("TASK_ID", "")
BROWSER_OUTPUT_PATH = TENSORBOARD_DIR.replace("tensorboard", "browser_config") + "/" + TASK_ID + "/" if TENSORBOARD_DIR else ""

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def get_ws_endpoint_list():
    # [MODIFIED] - 全局 ws_endpoint_list 在模块加载时准备好
    ws_endpoint_list = []
    try:
        browser_config_files = []
        #每个节点都有浏览器
        while len(browser_config_files) < int(NUM_NODES):
            browser_config_dir = BROWSER_OUTPUT_PATH
            browser_config_files = glob.glob(os.path.join(browser_config_dir, "*.json"))
            logger.info(f"Found {len(browser_config_files)} browser config files in {browser_config_files} {browser_config_dir}.")
        for file_path in browser_config_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded browser config from {file_path}: {data}")
            ws_endpoint_list.extend(data)
        logger.info(f"Loaded {len(ws_endpoint_list)} browser WebSocket endpoints.")
    except Exception as e:
        logger.error(f"Failed to load browser WebSocket endpoints: {e}", exc_info=True)
    return ws_endpoint_list

def get_storage_state_from_config_file(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return get_storage_state_from_start_url(config['start_url'])

def get_storage_state_from_start_url(start_url):
    local_storage_state_path = os.environ.get("LOCAL_STORAGE_STATE_PATH", "")
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
        storage_state: StorageState = {
            "cookies": [],
            "origins": []
        }
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
    return np.array(Image.open(BytesIO(png)).convert("RGB"))


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




def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def draw_image_with_coords(image: Image.Image, coords: Tuple[int, int]):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    x, y = coords
    radius = 15
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="red", width=3)
    draw.line([x - radius/2, y, x + radius/2, y], fill="red", width=2)
    draw.line([x, y - radius/2, x, y + radius/2], fill="red", width=2)
    return img

def calculate_ssim(imageA, imageB) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) between two images.
    Accepts both PIL Images and NumPy arrays as input.
    """
    # 1. 统一输入为NumPy数组
    if isinstance(imageA, Image.Image):
        imageA = np.array(imageA)
    if isinstance(imageB, Image.Image):
        imageB = np.array(imageB)

    # 2. 确保尺寸相同 (NumPy数组的 .shape)
    if imageA.shape != imageB.shape:
        # 如果尺寸不同，需要用PIL或OpenCV来resize，这里用PIL为例
        h, w = imageA.shape[:2]
        imageB_pil = Image.fromarray(imageB)
        imageB_resized_pil = imageB_pil.resize((w, h), Image.LANCZOS)
        imageB = np.array(imageB_resized_pil)

    # 3. 统一转换为灰度图
    # 检查图像是否已经是灰度图 (2D array)
    if len(imageA.shape) > 2:
        # 如果是彩色图 (3D array), 转换为灰度
        # 使用标准的亮度转换公式 (更精确) 或简单的平均
        grayA = np.dot(imageA[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        grayA = imageA

    if len(imageB.shape) > 2:
        grayB = np.dot(imageB[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        grayB = imageB

    # 4. 计算SSIM
    (score, diff) = ssim(grayA, grayB, full=True, data_range=255)
    
    return score

def extract_coords_by_index(solution_str: str, index: int) -> Optional[Tuple[int, int]]:
    """
    从 solution_str 中解析第index次工具调用，并提取其 'coords' 参数。

    Args:
        solution_str: 包含整个对话历史和工具调用的字符串。

    Returns:
        一个包含 (x, y) 坐标的元组，如果找不到或解析失败则返回 None。
    """

    last_tool_call_str = extract_tool_call_by_index(solution_str, index=index)
    try:
        # 解析JSON内容
        tool_call_data = json_repair.loads(last_tool_call_str)
        # 检查 'arguments' 和 'coords' 是否存在
        arguments = tool_call_data.get("arguments", {})
        if not isinstance(arguments, dict):
                logger.warning(f"BboxJudgeEvaluator: 'arguments' 格式不正确。工具调用: {last_tool_call_str}")
                return None

        coords = arguments.get("coords")
        if coords and isinstance(coords, list) and len(coords) == 2:
            # 假设 coords 是 [x, y]，并转换为整数
            x, y = int(coords[0]), int(coords[1])
            return x, y
        else:
            logger.warning(f"BboxJudgeEvaluator: 第{index}次工具调用中没有有效的 'coords'。工具调用: {last_tool_call_str}")
            return None
    except (Exception) as e:
        # 捕获JSON解析错误或类型转换错误
        logger.error(f"BboxJudgeEvaluator: 解析第{index}次工具调用时出错: {e}。工具调用字符串: '{last_tool_call_str}'")
        return None


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def extract_tool_call_arguments(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def extract_answer(solution_str):
    """The scoring function for exact match (EM) with format reward.

    Args:
        solution_str: the solution text
    
    Returns:
        float: Total reward score (format reward + answer reward)
    """
    if solution_str is None:
        return None
    answer = None
    try:
        # Extract answer from <answer> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        #logger.info(f"[INFO] Extract Answer: {answer}")
        return answer
    except Exception as e:
        solution_str_f = solution_str.replace("<|image_pad|>", "")
        #logger.error(f"[ERROR] Error in extract answer: {e}")
        return None

    return answer

def extract_tool_call_by_index(solution_str, index=0):
    """The scoring function for exact match (EM) with format reward.

    Args:
        solution_str: the solution text
    
    Returns:
        float: Total reward score (format reward + answer reward)
    """
    if solution_str is None:
        return None
    tool_call = None
    try:
        # Extract tool_call from <tool_call> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[index]
        tool_call = extract_tool_call_arguments(solution_str)
        #logger.info(f"[INFO] Extract Tool Call: {tool_call}")
        return tool_call
    except Exception as e:
        logger.error(f"[ERROR] Error in extract tool_call: {e}, solution_str: {solution_str}")
        return None
    return tool_call

def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    MAX_RATIO = 100
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def function_to_tool(func: Callable) -> Tool:
    """
    Convert a Python function to a Tool object using transformers.utils.get_json_schema.
    
    The function must have proper type annotations for all parameters and Google-style
    docstrings for the function description and parameter descriptions.
    
    Args:
        func: The Python function to convert to a tool. Must have:
            1. Type annotations for all parameters
            2. Google-style docstring with function description and parameter descriptions
            3. For enum parameters, add (choices: ["value1", "value2"]) at the end of the parameter description
    
    Returns:
        A Tool instance that wraps the provided function
    """
    # Get the JSON schema for the function
    schema = get_json_schema(func)
    
    # Extract the relevant information
    function_data = schema.get("function", {})
    name = function_data.get("name", func.__name__)
    description = function_data.get("description", "")
    parameters = function_data.get("parameters", {})
    
    # Create a tool class for this function
    class FunctionTool(Tool):
        def __init__(self):
            super().__init__(name=name, description=description, parameters=parameters)
            self.func = func
        
        def execute(self, args: Dict[str, Any]) -> str:
            """
            Execute the wrapped function with the provided arguments
            
            Args:
                args: Arguments to pass to the function
                
            Returns:
                Result of the function execution as a string
            """
            # Filter args to only include parameters that exist in the function signature
            sig = inspect.signature(self.func)
            valid_args = {k: v for k, v in args.items() if k in sig.parameters}
            
            try:
                result = self.func(**valid_args)
                # Convert result to string if it's not already
                if not isinstance(result, str):
                    result = str(result)
                return result
            except Exception as e:
                return f"Error executing {self.name}: {str(e)}"
    
    # Return an instance of the new tool class
    return FunctionTool()


# Example usage of function_to_tool:
#
# def search_weather(city: str, units: str = "metric"):
#     """
#     Search for weather information for a city.
#     
#     Args:
#         city: The name of the city to search for
#         units: The units to use for temperature (choices: ["metric", "imperial"])
#     
#     Returns:
#         Weather information for the specified city
#     """
#     # Implementation...
#     
# weather_tool = function_to_tool(search_weather)


def tool_decorator(name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to convert a function into a Tool object.
    
    Args:
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description (defaults to function docstring)
        
    Returns:
        A decorator function that converts the decorated function to a Tool
    """
    def decorator(func: Callable) -> Tool:
        tool = function_to_tool(func)
        
        # Override name and description if provided
        if name is not None:
            tool.name = name
        if description is not None:
            tool.description = description
            
        return tool
    
    return decorator


# Example usage of tool_decorator:
#
# @tool_decorator(name="GetWeather")
# def search_weather(city: str, units: str = "metric"):
#     """
#     Search for weather information for a city.
#     
#     Args:
#         city: The name of the city to search for
#         units: The units to use for temperature (choices: ["metric", "imperial"])
#     
#     Returns:
#         Weather information for the specified city
#     """
#     # Implementation... 


