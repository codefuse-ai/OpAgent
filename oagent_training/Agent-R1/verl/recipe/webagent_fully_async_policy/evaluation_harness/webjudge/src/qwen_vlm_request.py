from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import mimetypes
import base64
import os
from openai import OpenAI, APIError
import httpx
import json,re
import time
import requests
import io
import numpy as np
from PIL import Image
from recipe.webagent_fully_async_policy.evaluation_harness.webjudge.src.utils import encode_image_to_base64, encode_image
from recipe.webagent_fully_async_policy.constant.ak import API_KEYS_ANTCHAT, API_KEYS_MATRIX_MAPPED
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

FIXED_SEED=42
@dataclass
class Usage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

@dataclass
class Message:
    role: str = ''
    content: str = ''
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None

@dataclass
class Choice:
    finish_reason: str = ''
    index: int = -9
    message: Message = field(default_factory=lambda: Message(role='', content=''))
    logprobs: Optional[Dict[str, Any]] = None

@dataclass
class ChatCompletion:
    id: str = ''
    created: int = -9
    model: str = ''
    object: str = ''
    choices: List[Choice] = field(default_factory=lambda: [Choice()])
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None


# 将字典转换为 ChatCompletion 对象
def dict_to_chat_completion(data: dict) -> ChatCompletion:
    choices = []
    if data.get('choices', None):
      for choice_data in data.get('choices'):
          message_data = choice_data.get('message')
          message = Message(
              role=message_data.get('role'),
              content=message_data.get('content'),
              tool_calls=message_data.get('tool_calls'),
              function_call=message_data.get('function_call')
          )
          choice = Choice(
              finish_reason=choice_data.get('finish_reason'),
              index=choice_data.get('index'),
              message=message,
              logprobs=choice_data.get('logprobs')
          )
          choices.append(choice)
    else:
        choices = [Choice()]
    
    usage_data = data.get('usage')
    usage = Usage(
        completion_tokens=usage_data.get('completion_tokens'),
        prompt_tokens=usage_data.get('prompt_tokens'),
        total_tokens=usage_data.get('total_tokens')
    ) if usage_data else None
    
    return ChatCompletion(
        id=data.get('id'),
        created=data.get('created'),
        model=data.get('model'),
        object=data.get('object'),
        choices=choices,
        system_fingerprint=data.get('system_fingerprint'),
        usage=usage
    )

class ChatResponse:
    def __init__(self, choices, usage):
        self.choices = choices
        self.usage = usage


# 辅助函数：将 openai 库的响应对象转换为你自定义的 dataclass 对象
def _convert_openai_response_to_custom_chat_completion(openai_completion) -> ChatCompletion:
    """
    将 openai.types.chat.ChatCompletion 对象转换为我们自定义的 ChatCompletion dataclass。
    """
    if not openai_completion:
        return ChatCompletion()

    # 转换 choices
    custom_choices = []
    if openai_completion.choices:
        for choice in openai_completion.choices:
            # 转换 message
            # openai v1.x+ message.tool_calls is a list of objects, not dicts. We convert them.
            tool_calls_as_dicts = [tc.model_dump() for tc in choice.message.tool_calls] if choice.message.tool_calls else None
            # function_call is also an object
            function_call_as_dict = choice.message.function_call.model_dump() if choice.message.function_call else None

            custom_message = Message(
                role=choice.message.role,
                content=choice.message.content,
                tool_calls=tool_calls_as_dicts,
                function_call=function_call_as_dict
            )
            custom_choice = Choice(
                finish_reason=choice.finish_reason,
                index=choice.index,
                message=custom_message,
                logprobs=choice.logprobs # logprobs is usually None unless requested
            )
            custom_choices.append(custom_choice)
    else:
        custom_choices = [Choice()]

    # 转换 usage
    custom_usage = None
    if openai_completion.usage:
        custom_usage = Usage(
            completion_tokens=openai_completion.usage.completion_tokens,
            prompt_tokens=openai_completion.usage.prompt_tokens,
            total_tokens=openai_completion.usage.total_tokens
        )

    # 构建并返回最终的自定义 ChatCompletion 对象
    return ChatCompletion(
        id=openai_completion.id,
        created=openai_completion.created,
        model=openai_completion.model,
        object=openai_completion.object,
        choices=custom_choices,
        system_fingerprint=openai_completion.system_fingerprint,
        usage=custom_usage
    )


# ---------------------------------------------------------------------------
# 2. 这是你需要的、带有显式重试逻辑的新函数实现
# ---------------------------------------------------------------------------
def send_chat_completion_request(messages, model="Qwen2.5-VL-72B-Instruct", max_retries=10, retry_delay=10, temperature=0.1):
    """
    使用 openai 库发送请求，并实现了显式的、可自定义的重试逻辑。
    支持多个API key的循环调用。
    """
    base_url = os.environ.get("ANTCHAT_ENDPOINT", "http://localhost:8000/v1")

    # 将两个API key放入list中，支持循环调用
    api_keys = API_KEYS_ANTCHAT if API_KEYS_ANTCHAT else [os.environ.get("ANTCHAT_API_KEY", "")]
    
    attempts = 0
    while attempts < max_retries:

        # 选择当前尝试使用的API key（循环使用）
        current_api_key = api_keys[attempts % len(api_keys)]

        with OpenAI(
            api_key=current_api_key,
            base_url=base_url,
            max_retries=max_retries, 
            timeout=10 * 60.0,  # 设置一个合理的请求超时时间
        ) as client:

            try:
                logger.info(f"尝试第 {attempts + 1}/{max_retries} 次请求，使用API key: {current_api_key[:8]}...")
                
                completion_from_openai = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=FIXED_SEED
                )
                # 如果请求成功，转换响应并立即返回
                return _convert_openai_response_to_custom_chat_completion(completion_from_openai)

            except APIError as e:
                attempts += 1
                logger.error(f"API 请求失败 (第 {attempts}/{max_retries} 次尝试，key: {current_api_key[:8]}): {e}")
                if attempts < max_retries:
                    logger.info(f"将在 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error("已达到最大重试次数，请求最终失败。")
            except Exception as e:
                # 捕获其他可能的错误，如网络连接问题
                attempts += 1
                logger.error(f"发生未知错误 (第 {attempts}/{max_retries} 次尝试，key: {current_api_key[:8]}): {e}")
                if attempts < max_retries:
                    logger.info(f"将在 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error("已达到最大重试次数，请求最终失败。")

    # 如果循环结束仍未成功返回，则返回一个空的 ChatCompletion 对象
    return ChatCompletion()

def extractJson(input_str):
    # 提取代码块标记的正则表达式
    regex = r"```([\s\S]*?)```"
    pattern = re.compile(regex)
    
    json_input = input_str
    # 取最新的匹配
    for match in pattern.finditer(input_str):
        json_input = match.group(1)
    
    # 去除起始和结束的```json和```，特殊字符转义，json末尾的中文双引号不稳定问题
    cleaned = json_input.replace("```JSON", "") \
                        .replace("```json", "") \
                        .replace("```", "") \
                        .replace("json\n", "") \
                        .replace("JSON\n", "") \
                        .replace("json", "") \
                        .replace("JSON", "") \
                        .replace("”\n}", "\"}")
    
    # 去除前后可能存在的空白字符
    cleaned = cleaned.strip()
    return cleaned

# 假设 getJsonObject 是一个解析 JSON 字符串的函数
def getJsonObject(llm_answer):
    try:
        llm_answer = extractJson(llm_answer)
        return json.loads(llm_answer)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return None

def send_chat_completion_request_shangshu(messages, model="qwen-vl-max-2025-08-13", max_retries=10, retry_delay=10, 
    temperature=0.1, max_tokens=None):
    """
    使用 openai 库发送请求，并实现了显式的、可自定义的重试逻辑。
    支持多个API key的循环调用。
    """
    import os

    # 在脚本开头清除代理环境变量
    if 'HTTP_PROXY' in os.environ:
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        del os.environ['HTTPS_PROXY']

    base_url = os.environ.get("MATRIXLLM_ENDPOINT", "http://localhost:8000/v1")
    # 将两个API key放入list中，支持循环调用
    api_keys = API_KEYS_MATRIX_MAPPED.get("qwen", [os.environ.get("MATRIXLLM_API_KEY", "")])
    
    model="qwen-vl-max"
    
    attempts = 0
    while attempts < max_retries:

        # 选择当前尝试使用的API key（循环使用）
        current_api_key = api_keys[attempts % len(api_keys)]
        
        with OpenAI(
            api_key=current_api_key,
            base_url=base_url,
            max_retries=max_retries, 
            timeout=10 * 60.0  # 设置一个合理的请求超时时间
        ) as client:

            try:
                logger.info(f"尝试第 {attempts + 1}/{max_retries} 次请求，使用API key: {current_api_key[:8]}...")
                
                if max_tokens is not None:
                    completion_from_openai = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        #top_p = 1.0,
                        #extra_body={"top_k":0},
                        #response_format={"type": "json_object"},
                        #top_k = 0
                        seed=FIXED_SEED,
                        max_tokens=max_tokens,
                    )
                else:
                    completion_from_openai = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        #top_p = 1.0,
                        #extra_body={"top_k":0},
                        #response_format={"type": "json_object"},
                        #top_k = 0
                        seed=FIXED_SEED,
                    )
                request_id = completion_from_openai.id
                logger.info(f"Request ID: {request_id}")
                # 如果请求成功，转换响应并立即返回
                return _convert_openai_response_to_custom_chat_completion(completion_from_openai)

            except APIError as e:
                attempts += 1
                logger.error(f"API 请求失败 (第 {attempts}/{max_retries} 次尝试，key: {current_api_key[:8]}): {e}")
                if attempts < max_retries:
                    logger.info(f"将在 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error("已达到最大重试次数，请求最终失败。")
            except Exception as e:
                # 捕获其他可能的错误，如网络连接问题
                attempts += 1
                logger.error(f"发生未知错误 (第 {attempts}/{max_retries} 次尝试，key: {current_api_key[:8]}): {e}")
                if attempts < max_retries:
                    logger.info(f"将在 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error("已达到最大重试次数，请求最终失败。")

    # 如果循环结束仍未成功返回，则返回一个空的 ChatCompletion 对象
    return ChatCompletion()

# ---------------------------------------------------------------------------
# 3. 新增的图像处理和测试样本
# ---------------------------------------------------------------------------

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """读取本地图像文件并将其编码为 Base64 Data URI"""
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"错误: 图像文件未找到 at path: {image_path}")
            return None
            
        # 猜测图像的 MIME 类型 (e.g., 'image/png', 'image/jpeg')
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "application/octet-stream" # 如果无法确定，则使用通用类型

        # 读取二进制文件内容
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
        
        # Base64 编码
        base64_encoded_data = base64.b64encode(binary_data)
        base64_string = base64_encoded_data.decode('utf-8')

        # 格式化为 Data URI
        return f"data:{mime_type};base64,{base64_string}"
    except Exception as e:
        logger.error(f"编码图像时出错: {e}")
        return None


def compare_two_images_with_shangshu(
    image1,
    image2,
    user_query,
    max_tokens=None):
    whole_content_img = []
    if isinstance(image1, np.ndarray):
        image1 = Image.fromarray(image1)
    if isinstance(image2, np.ndarray):
        image2 = Image.fromarray(image2)
    jpg_base64_str_1 = encode_image(image1, max_size=1024)
    whole_content_img.append(
        {
            'type': 'image_url',
            'image_url': {"url": f"data:image/png;base64,{jpg_base64_str_1}", "detail": "high"}
        }
    )
    jpg_base64_str_2 = encode_image(image2, max_size=1024)
    whole_content_img.append(
        {
            'type': 'image_url',
            'image_url': {"url": f"data:image/png;base64,{jpg_base64_str_2}", "detail": "high"}
        }
    )
    text_prompt = '''
# ROLE:
You are an expert web automation assistant. Your task is to determine if an action performed on a webpage resulted in positive progress towards a user's objective.

# CONTEXT:
You will be given three pieces of information:
1.  **Objective**: The user's ultimate goal.
2.  **Image Before**: A screenshot of the webpage *before* the action was taken.
3.  **Image After**: A screenshot of the webpage *after* the action was taken.

# INSTRUCTIONS:
Analyze the change between the "Image Before" and "Image After" in the context of the user's "Objective".

"Positive progress" (a "Yes" answer) includes, but is not limited to:
- Revealing new, relevant information or controls (e.g., a login form appearing, search results loading).
- Successfully completing an input (e.g., text appearing in a username field).
- Removing an obstacle (e.g., closing a cookie banner, ad, or pop-up).
- Navigating to a page that is logically closer to the objective.
- Making a necessary intermediate step (e.g., clicking "agree" on terms and conditions).

"No progress or negative progress" (a "No" answer) includes:
- No meaningful visual change.
- Navigating to an irrelevant page (e.g., "About Us" page when trying to buy a product).
- Encountering an error message.
- Being stuck in the same state.
- Introducing a new, irrelevant obstacle (e.g., an ad appearing).

First, provide a brief, step-by-step analysis of your reasoning. Then, on a new line, provide your final answer in the specified format.

# ANALYSIS FORMAT:
1.  **Objective Analysis**: Briefly state what the user is trying to achieve.
2.  **Visual Analysis**: Describe the key visual difference between the 'Before' and 'After' images.
3.  **Progress Analysis**: Explain whether this change is helpful for the objective and why.

# FINAL ANSWER FORMAT:
Final Answer: [Yes/No]

---
# INPUT:

**Objective**: {user_query}

**Image Before**:
<image>

**Image After**:
<image>
    '''.format(user_query=user_query)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text_prompt}] + whole_content_img
        }
    ]
    response = send_chat_completion_request_shangshu(messages, max_tokens=max_tokens)
    return response.choices[0].message.content

def describe_status_image_with_shangshu(
    pil_image, 
    user_query,
    max_tokens=None):
    #将PIL 格式的图片转换为Base64 Data URI
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    base64_encoded_data = base64.b64encode(buffered.getvalue())
    base64_image = base64_encoded_data.decode('utf-8')
    base64_image = f"data:image/jpeg;base64,{base64_image}"
    text_prompt = '''
# ROLE & GOAL
You are a highly specialized Vision Analysis Agent, a component within a larger web automation system. Your sole function is to analyze a webpage screenshot and generate a concise, factual description of its current state. This description is mission-critical for the master agent to determine the next logical action.

# CONTEXT
You will receive two inputs:
1.  `user_goal`: A string describing the user's ultimate objective (e.g., "log into my account", "find a flight to Beijing", "buy a mechanical keyboard").
2.  `screenshot`: An image of the current browser viewport.

Your goal is to describe the current state of the webpage, focusing **exclusively** on elements that are relevant to achieving the `user_goal`.

# INSTRUCTIONS
1.  **Analyze the Screenshot:** Scrutinize the image to identify all visible UI elements, text, and overall page structure.
2.  **Prioritize by Relevance:** Evaluate each element's relevance to the `user_goal`. Your description must prioritize interactive elements (e.g., buttons, input fields, links, dropdowns) and key information (e.g., status messages, error notifications, search results, product details) that are directly on the path to fulfilling the user's goal.
3.  **Be Strictly Observational:** Describe only what you *see*. Do not infer user intent, predict outcomes, or suggest actions. For example, describe "a login form with a username field" NOT "the user should now enter their username".
4.  **Aggressively Ignore Irrelevance:** Actively discard and do not mention generic elements such as common headers/footers, unrelated advertisements, decorative images, or any content not pertinent to the user's immediate task. This is essential for conciseness.

# OUTPUT REQUIREMENTS
1.  **Word Count:** The entire description must be **under 100 words**. Brevity is paramount.
2.  **Language:** English.
3.  **Focus:** The output must be laser-focused on goal-relevant elements and the overall page state.
4.  **Format:** Plain text only. Do not include any preamble (e.g., "Here is the description:") or any other conversational filler.

---

### EXAMPLES

**Example 1:**
*   **`user_goal`**: "log into my GitHub account"
*   **`[Screenshot]`**: An image of the GitHub login page.
*   **`Expected Output`**: The page displays the GitHub login form. It contains two input fields labeled 'Username or email address' and 'Password', and a green 'Sign in' button. The input fields are currently empty. There are also links for 'Forgot password?' and 'Create an account'.

**Example 2:**
*   **`user_goal`**: "search for 'drone' on Amazon"
*   **`[Screenshot]`**: An image of Amazon's search results for "drone".
*   **`Expected Output`**: The page shows a list of search results for 'drone'. Multiple products are displayed in a grid, each with an image, title, star rating, and price. Filtering and sorting options like 'Brand' and 'Price' are visible near the top. The page appears fully loaded and ready for interaction.

**Example 3:**
*   **`user_goal`**: "book a flight from Shanghai to Beijing for tomorrow"
*   **`[Screenshot]`**: An image of a flight booking website showing an error message.
*   **`Expected Output`**: An error is displayed on the flight booking form. Red text below the date selection field reads 'Return date cannot be earlier than departure date'. The origin 'Shanghai' and destination 'Beijing' are correctly filled. The return date needs to be corrected to proceed.

---

### FINAL PROMPT TEMPLATE FOR EXECUTION

(This is the condensed template you would use in your application's API call.)

ROLE
You are a vision-analysis agent. Your task is to describe a webpage screenshot, focusing only on elements relevant to the user's goal. Your description must be under 100 words and be a plain text observation.

USER GOAL
{user_query}

INSTRUCTIONS
Analyze the following screenshot and describe the current page state in relation to the user's goal. Focus on interactive elements, relevant information, and status messages. Ignore all irrelevant content.
    '''.format(user_query=user_query)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # 这里放入 Base64 编码后的 Data URI
                        "url": base64_image
                    }
                }
            ]
        }
    ]
    response = send_chat_completion_request_shangshu(messages, max_tokens=max_tokens)
    return response.choices[0].message.content

def describe_image_with_shangshu(pil_image, 
    text_prompt="Please describe the image in detail. No more than 50 words.", 
    max_tokens=None):
    #将PIL 格式的图片转换为Base64 Data URI
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    base64_encoded_data = base64.b64encode(buffered.getvalue())
    base64_image = base64_encoded_data.decode('utf-8')
    base64_image = f"data:image/jpeg;base64,{base64_image}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # 这里放入 Base64 编码后的 Data URI
                        "url": base64_image
                    }
                }
            ]
        }
    ]
    response = send_chat_completion_request_shangshu(messages, max_tokens=max_tokens)
    return response.choices[0].message.content


# --- UPDATED GENERIC FUNCTION for all MatrixLLM models with Key Rotation ---
def send_chat_completion_request_matrix(messages, model, max_retries=10, retry_delay=10, temperature=0.1, max_tokens=None):
    """
    Sends a request to the MatrixLLM Pool endpoint.
    **NEW: Selects a LIST of keys based on the model name and rotates through them on retries.**
    """
    if 'HTTP_PROXY' in os.environ: del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ: del os.environ['HTTPS_PROXY']
    
    base_url = os.environ.get("MATRIXLLM_ENDPOINT", "http://localhost:8000/v1")

    # --- KEY LIST SELECTION LOGIC ---
    selected_key_list = None
    model_name_lower = model.lower()
    
    if "qwen" in model_name_lower:
        selected_key_list = API_KEYS_MATRIX_MAPPED.get("qwen")
    elif "gemini" in model_name_lower:
        selected_key_list = API_KEYS_MATRIX_MAPPED.get("gemini")
    
    # Fallback to the default list of keys
    if not selected_key_list:
        logger.warning(f"Warning: No specific key list for model '{model}'. Using default key list.")
        selected_key_list = API_KEYS_MATRIX_MAPPED.get("default")

    if not selected_key_list or not isinstance(selected_key_list, list) or len(selected_key_list) == 0:
        logger.error(f"FATAL: No valid API key list found for model '{model}'. Aborting request.")
        return ChatCompletion()
    # --- END OF KEY LIST SELECTION LOGIC ---

    attempts = 0
    while attempts < max_retries:
        # **NEW: Rotate through the selected list of keys on each attempt**
        current_key = selected_key_list[attempts % len(selected_key_list)]
        
        with OpenAI(api_key=current_key, base_url=base_url, max_retries=0, timeout=10 * 60.0) as client:
            try:
                logger.info(f"Attempt {attempts + 1}/{max_retries} for model '{model}' with key: {current_key[:8]}... (from list of {len(selected_key_list)})")
                args = {"model": model, "messages": messages, "temperature": temperature, "seed": FIXED_SEED}
                if max_tokens: args["max_tokens"] = max_tokens
                completion_from_openai = client.chat.completions.create(**args)
                return _convert_openai_response_to_custom_chat_completion(completion_from_openai)
            except APIError as e:
                attempts += 1
                logger.error(f"API Request Failed: {e}")
                if attempts < max_retries: logger.info(f"Retrying in {retry_delay} seconds..."); time.sleep(retry_delay)
                else: logger.error("Maximum retries reached. Request failed.")
            except Exception as e:
                attempts += 1
                logger.error(f"An unexpected error occurred: {e}")
                if attempts < max_retries: logger.info(f"Retrying in {retry_delay} seconds..."); time.sleep(retry_delay)
                else: logger.error("Maximum retries reached. Request failed.")
    return ChatCompletion()