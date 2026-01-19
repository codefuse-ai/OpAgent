"""
Utility functions for working with tools
"""
import re
import urllib
import importlib
import inspect
from typing import Callable, Dict, Any, Type, Optional, Tuple
from functools import wraps
import os
import json_repair
import logging
from PIL import Image, ImageDraw
import numpy as np
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

# from agent_r1.tool.tool_base import Tool
class Tool: pass
import math

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
    # logger.debug(f"!!!name BboxJudgeEvaluator: last_tool_call_str: {last_tool_call_str}")
    try:
        # 解析JSON内容
        tool_call_data = json_repair.loads(last_tool_call_str)
        # logger.debug(f"!!!name BboxJudgeEvaluator: tool_call_data: {tool_call_data}")
        # 检查 'arguments' 和 'coords' 是否存在
        arguments = tool_call_data.get("arguments", {})
        if not isinstance(arguments, dict):
                logger.warning(f"!!!name BboxJudgeEvaluator: 'arguments' 格式不正确。工具调用: {last_tool_call_str}")
                return None

        coords = arguments.get("coords")
        # logger.debug(f"!!!name BboxJudgeEvaluator: coords: {coords}")
        if coords and isinstance(coords, list) and len(coords) == 2:
            # 假设 coords 是 [x, y]，并转换为整数
            x, y = int(coords[0]), int(coords[1])
            return x, y
        else:
            logger.warning(f"!!!name BboxJudgeEvaluator: 第{index}次工具调用中没有有效的 'coords'。工具调用: {last_tool_call_str}")
            return None
    except (Exception) as e:
        # 捕获JSON解析错误或类型转换错误
        logger.error(f"!!!name BboxJudgeEvaluator: 解析第{index}次工具调用时出错: {e}。工具调用字符串: '{last_tool_call_str}'")
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