#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Interface - vLLM Local Model API

Uses vLLM to load codefuse-ai/OpAgent (based on Qwen3-32B) model.
Reference: inference.py implementation.

Model output format:
<think>Thinking process...</think>
<tool_call>{"name": "click", "arguments": {"coords": [x, y]}}</tool_call>
"""

import json
import base64
import re
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from io import BytesIO
from loguru import logger

from PIL import Image
from PIL.Image import Image as ImageObject

# vLLM imports
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_CONFIG = {
    "model_path": "codefuse-ai/OpAgent",  # HuggingFace model path or local path
    "tensor_parallel_size": 1,  # GPU parallelism, adjust based on your GPU count
    "gpu_memory_utilization": 0.9,  # GPU memory utilization, default 0.9 (90%)
    "max_model_len": 32768,  # Limit max sequence length to avoid KV cache OOM
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 0.001,
    "repetition_penalty": 1.05,
    "max_pixels": 2981888,
    "min_pixels": 262144,
}

# Global model instances
_llm: Optional[LLM] = None
_processor = None
_sampling_params: Optional[SamplingParams] = None


# =============================================================================
# Tool Definitions (consistent with prompt.py)
# =============================================================================

TOOL_LISTS = [
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


@dataclass
class ModelInput:
    """Model input data structure"""
    screenshot_base64: str  # Base64 encoded screenshot
    query: str  # User query
    history: List[Dict[str, Any]]  # History of actions
    current_url: str  # Current page URL


@dataclass
class ModelOutput:
    """Model output data structure"""
    raw_response: str  # Raw response
    think_content: Optional[str]  # <think> content
    tool_call: Optional[Dict[str, Any]]  # Parsed tool_call (unified format)


# Scaling functions consistent with qwen_vl_utils
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 784, max_pixels: int = 3211264) -> Tuple[int, int]:
    """
    Calculate the compressed image size for model input (consistent with qwen_vl_utils internal logic)
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar


def process_image(image: Union[bytes, str, ImageObject], max_pixels: int, min_pixels: int) -> Tuple[ImageObject, float, float, Tuple[int, int], Tuple[int, int]]:
    """
    Process image, calculate scaling ratio
    
    Returns:
        (image, width_revert, height_revert, original_size, resized_size)
        - width_revert, height_revert: Coordinate revert factors (model coords * revert = original coords)
        - original_size: (width, height) original dimensions
        - resized_size: (width, height) model input dimensions
    """
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        if image.startswith('data:'):
            base64_data = image.split(',')[1] if ',' in image else image
            image = Image.open(BytesIO(base64.b64decode(base64_data)))
        else:
            image = Image.open(image)
    
    original_width, original_height = image.width, image.height
    
    # Use smart_resize consistent with qwen_vl_utils to calculate compressed size (multiple of 28)
    resized_height, resized_width = smart_resize(original_height, original_width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    # Calculate coordinate revert factors (model coords * revert = original coords)
    width_revert = original_width / resized_width
    height_revert = original_height / resized_height
    
    # Resize image to multiple of 28 (so process_vision_info output size matches)
    if (resized_width, resized_height) != (original_width, original_height):
        image = image.resize((resized_width, resized_height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image, width_revert, height_revert, (original_width, original_height), (resized_width, resized_height)


def load_model():
    """Load vLLM model and processor
    
    GPU Configuration Notes:
    1. tensor_parallel_size: Load model across multiple GPUs in parallel
    2. gpu_memory_utilization: GPU memory usage ratio per GPU
    3. Use CUDA_VISIBLE_DEVICES environment variable to specify GPUs
       Example: export CUDA_VISIBLE_DEVICES=0,1,2,3
    """
    global _llm, _processor, _sampling_params
    
    if _llm is not None:
        return _llm, _processor, _sampling_params
    
    import os
    import torch
    
    # Get GPU info
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    model_path = MODEL_CONFIG["model_path"]
    tensor_parallel_size = MODEL_CONFIG["tensor_parallel_size"]
    gpu_memory_utilization = MODEL_CONFIG["gpu_memory_utilization"]
    
    logger.info(f"🔄 Loading model from: {model_path}")
    logger.info(f"   Available GPUs: {available_gpus}")
    logger.info(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    logger.info(f"   Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"   GPU memory utilization: {gpu_memory_utilization}")
    
    # Validate configuration
    if tensor_parallel_size > available_gpus:
        logger.warning(f"⚠️  tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({available_gpus})")
        logger.warning(f"   Adjusting to {available_gpus}")
        tensor_parallel_size = max(1, available_gpus)
    
    # Load vLLM model
    max_model_len = MODEL_CONFIG.get("max_model_len", 32768)
    logger.info(f"   Max model len: {max_model_len}")
    
    _llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,  # Limit max sequence length to avoid KV cache OOM
        limit_mm_per_prompt={"image": 10, "video": 1},  # Support multiple images
        trust_remote_code=True,
    )
    
    # Load processor
    _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Create sampling parameters
    _sampling_params = SamplingParams(
        temperature=MODEL_CONFIG["temperature"],
        top_p=MODEL_CONFIG["top_p"],
        repetition_penalty=MODEL_CONFIG["repetition_penalty"],
        max_tokens=MODEL_CONFIG["max_new_tokens"],
        stop_token_ids=[],
    )
    
    logger.info(f"✅ Model loaded successfully")
    
    return _llm, _processor, _sampling_params


def build_system_prompt() -> str:
    """Build system prompt"""
    system_prompt = """You are an excellent web agent. Based on the web screen shot and content, your need call the single, most appropriate tool for the current step to make progress on the user's request."""
    
    system_prompt += """Output the thinking process in <think> </think> tags, and for each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<think> ... </think><tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>"""
    
    system_prompt += """\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"""
    
    tool_desc = json.dumps(TOOL_LISTS, ensure_ascii=False, indent=2)
    system_prompt += tool_desc + '\n</tools>'
    
    return system_prompt


def build_prompt(model_input: ModelInput, image: ImageObject, original_width: int, original_height: int) -> Tuple[str, List]:
    """
    Build prompt to send to model (reference inference.py format)
    
    Returns:
        (prompt_text, images_list)
    """
    instruction = model_input.query
    history = model_input.history
    
    # Process history actions
    step_num = min(len(history), 2)  # Keep at most 2 history steps
    
    prompt_parts = [
        "<image>\n", 
        f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n",
        f"Instruction: '{instruction}'.\n",
    ]
    prompt_str = "".join(prompt_parts)
    
    # Add history steps
    for i in range(step_num):
        h = history[-(step_num - i)]
        action_type = h.get('action_type', '')
        params = h.get('params', {})
        
        # Convert to model format (history pixel coords -> 0-1000 normalized coords)
        history_action = convert_to_model_format(action_type, params, original_width, original_height)
        history_thought = f"Executing {action_type} action"
        
        history_step = f"""Step {i+1}:\n<think>{history_thought}</think><tool_call>{json.dumps(history_action)}</tool_call>\n"""
        prompt_str += history_step
    
    prompt_str += f"""Step {step_num + 1}:"""
    
    # Image list (current screenshot)
    images = [image]
    
    return prompt_str, images


def convert_to_model_format(action_type: str, params: Dict, original_width: int = 1280, original_height: int = 800) -> Dict:
    """Convert action_executor format to model output format (consistent with prompt.py)
    
    Qwen3-VL outputs 0-1000 normalized coordinates
    Pixel coords -> Model coords: coords = pixel / original_size * 1000
    """
    result = {"name": action_type, "arguments": {}}
    
    # Get coordinates
    x = params.get('x') or params.get('coordinate_x')
    y = params.get('y') or params.get('coordinate_y')
    
    if x is not None and y is not None:
        # Pixel coords -> 0-1000 normalized coords
        model_x = int(x / original_width * 1000)
        model_y = int(y / original_height * 1000)
        result["arguments"]["coords"] = [model_x, model_y]
    
    if action_type == "type":
        result["arguments"]["content"] = params.get('text', params.get('content', ''))
        result["arguments"]["press_enter_after"] = 1 if params.get('press_enter', False) else 0
    elif action_type == "scroll":
        result["arguments"]["direction"] = params.get('direction', 'down')
        result["arguments"]["distance"] = params.get('distance', 300)
    elif action_type == "hscroll":
        result["arguments"]["direction"] = params.get('direction', 'right')
        result["arguments"]["distance"] = params.get('distance', 300)
    elif action_type == "press":
        result["arguments"]["key"] = params.get('key', 'Enter')
    elif action_type == "wait":
        result["arguments"]["seconds"] = params.get('seconds', 2)
    elif action_type == "goto":
        result["arguments"]["url"] = params.get('url', '')
    elif action_type == "tab_focus":
        result["arguments"]["tab_index"] = params.get('tab_index', 0)
    elif action_type == "answer":
        result["arguments"]["content"] = params.get('content', '')
    
    return result


def convert_from_model_format(model_output: Dict, original_width: int = 1280, original_height: int = 800) -> Dict:
    """Convert model output format to action_executor format (consistent with prompt.py)
    
    Qwen3-VL outputs 0-1000 normalized coordinates
    Model coords -> Pixel coords: pixel = coords / 1000 * original_size
    """
    name = model_output.get('name', '')
    arguments = model_output.get('arguments', {})
    
    result = {"action_type": name}
    
    # Terminal actions don't process coordinates
    terminal_actions = ["answer", "stop", "finish", "done"]
    
    # Process coordinates (only for non-terminal actions)
    if name not in terminal_actions:
        coords = arguments.get('coords')
        if coords and isinstance(coords, list) and len(coords) >= 2:
            # 0-1000 normalized coords -> Pixel coords
            result["x"] = round(coords[0] / 1000.0 * original_width)
            result["y"] = round(coords[1] / 1000.0 * original_height)
    
    # Process other parameters
    if name == "type":
        result["text"] = arguments.get('content', '')
        # Note: press_enter_after defaults to 0 in prompt.py
        result["press_enter"] = arguments.get('press_enter_after', 0) == 1
    elif name == "scroll":
        result["direction"] = arguments.get('direction', 'down')
        result["distance"] = arguments.get('distance', 300)
    elif name == "hscroll":
        result["direction"] = arguments.get('direction', 'right')
        result["distance"] = arguments.get('distance', 300)
    elif name == "press":
        result["key"] = arguments.get('key', 'Enter')
    elif name == "wait":
        result["seconds"] = arguments.get('seconds', 2)
    elif name == "hover":
        pass  # Coordinates already processed
    elif name == "move_to":
        # move_to is same as hover
        pass  # Coordinates already processed
    elif name == "goto":
        result["url"] = arguments.get('url', '')
    elif name == "tab_focus":
        result["tab_index"] = arguments.get('tab_index', 0)
    elif name in terminal_actions:
        result["action_type"] = "answer"
        result["content"] = arguments.get('content', '')
        # Ensure answer action has no coordinates
        result.pop('x', None)
        result.pop('y', None)
    
    return result


def parse_model_response(raw_response: str, original_width: int = 1280, original_height: int = 800) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Parse model response, extract think content and tool_call
    
    Qwen3-VL outputs 0-1000 normalized coordinates, need to convert to pixel coordinates
    
    Returns:
        (think_content, tool_call) - tool_call converted to action_executor format
    """
    think_content = None
    tool_call = None
    
    # Parse <think> content
    think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
    
    # Parse <tool_call> content
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', raw_response, re.DOTALL)
    if tool_call_match:
        try:
            tool_call_str = tool_call_match.group(1).strip()
            model_format = json.loads(tool_call_str)
            # Convert to action_executor format (0-1000 normalized coords -> pixel coords)
            tool_call = convert_from_model_format(model_format, original_width, original_height)
        except Exception as e:
            logger.warning(f"Failed to parse tool_call: {e}")
            # Try extracting JSON
            try:
                json_match = re.search(r'\{.*\}', tool_call_str, re.DOTALL)
                if json_match:
                    model_format = json.loads(json_match.group(0))
                    tool_call = convert_from_model_format(model_format, original_width, original_height)
            except Exception:
                pass
    
    return think_content, tool_call


def call_model(model_input: ModelInput) -> ModelOutput:
    """
    Call vLLM model to get next action
    
    Args:
        model_input: Contains screenshot, query, history, etc.
        
    Returns:
        ModelOutput: Contains think content and tool_call
    """
    logger.info(f"[Model] Calling OpAgent with query: {model_input.query[:50]}...")
    logger.info(f"[Model] History length: {len(model_input.history)}")
    logger.info(f"[Model] Current URL: {model_input.current_url}")
    
    try:
        # Load model
        llm, processor, sampling_params = load_model()
        
        # Decode and process image
        screenshot_bytes = base64.b64decode(model_input.screenshot_base64)
        image, width_revert, height_revert, original_size, resized_size = process_image(
            screenshot_bytes, 
            MODEL_CONFIG["max_pixels"], 
            MODEL_CONFIG["min_pixels"]
        )
        
        # Original size for coordinate conversion (Qwen3-VL outputs 0-1000 normalized coords)
        original_width, original_height = original_size
        
        logger.info(f"[Model] Original screenshot: {original_width}x{original_height} -> Model input: {resized_size[0]}x{resized_size[1]} (multiple of 28)")
        logger.info(f"[Model] Model coords 0-1000 will be converted to pixel coords (original size: {original_width}x{original_height})")
        
        # Build prompt (history coords use original size)
        prompt_text, images = build_prompt(model_input, image, original_width, original_height)
        system_prompt = build_system_prompt()
        
        # Build message format
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Process visual information
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        # Build vLLM input
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        
        llm_inputs = [{
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }]
        
        # Execute inference
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        raw_response = outputs[0].outputs[0].text
        
        logger.info(f"[Model] Response length: {len(raw_response)}")
        logger.debug(f"[Model] Raw response: {raw_response}")
        
    except Exception as e:
        logger.error(f"[Model] Model call failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error response
        raw_response = f'<think>Model call failed: {e}</think><tool_call>{{"name": "answer", "arguments": {{"content": "Model call failed, please check if model is loaded correctly"}}}}</tool_call>'
        original_width = 1280
        original_height = 800
    
    # Parse response (0-1000 normalized coords -> pixel coords)
    think_content, tool_call = parse_model_response(raw_response, original_width, original_height)
    
    if think_content:
        logger.info(f"[Model] Think: {think_content[:100]}...")
    
    if tool_call:
        logger.info(f"[Model] Tool call: {tool_call}")
    else:
        logger.warning("[Model] No valid tool_call found in response")
    
    return ModelOutput(
        raw_response=raw_response,
        think_content=think_content,
        tool_call=tool_call
    )


# =============================================================================
# Configuration Modification Functions
# =============================================================================

def set_model_config(model_path: str = None, 
                     tensor_parallel_size: int = None,
                     max_new_tokens: int = None,
                     temperature: float = None,
                     max_model_len: int = None,
                     gpu_memory_utilization: float = None):
    """
    Dynamically modify model configuration
    
    Example:
        set_model_config(
            model_path="/path/to/local/model",
            tensor_parallel_size=4,
            max_model_len=32768
        )
    """
    global _llm, _processor, _sampling_params
    
    if model_path is not None:
        MODEL_CONFIG["model_path"] = model_path
    if tensor_parallel_size is not None:
        MODEL_CONFIG["tensor_parallel_size"] = tensor_parallel_size
    if max_new_tokens is not None:
        MODEL_CONFIG["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        MODEL_CONFIG["temperature"] = temperature
    if max_model_len is not None:
        MODEL_CONFIG["max_model_len"] = max_model_len
    if gpu_memory_utilization is not None:
        MODEL_CONFIG["gpu_memory_utilization"] = gpu_memory_utilization
    
    # Reset model to use new configuration
    _llm = None
    _processor = None
    _sampling_params = None
    
    logger.info(f"[Model] Config updated: {MODEL_CONFIG}")


def get_model_config() -> Dict[str, Any]:
    """Get current model configuration"""
    return MODEL_CONFIG.copy()
