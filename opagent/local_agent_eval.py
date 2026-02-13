#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地 WebAgent 执行器 (Local Agent Evaluator)

基于 webarena_online_eval.py 的架构，但将远程 API 调用改为本地模型调用。

功能：
1. 从 ECS CSV 文件加载 ECS 实例列表
2. 使用 BrowserActor 管理浏览器连接
3. 每个任务执行前调用 ssh_connect_and_refreshweb 刷新网页状态
4. 本地调用 Reflector、Planner、Grounder 模型执行任务
5. 保存轨迹并进行评估

架构：
- 模型调用：参考 local_agent_eval_v2.py
- 浏览器操作：参考 replay_and_evaluate.py
- ECS 管理和刷新：参考 webarena_online_eval.py
"""

import os
import importlib.util
import sys

import sys
import json
import time
import asyncio
import base64
import traceback
import threading
import csv
import glob
import aiohttp
import re
import hashlib
import requests
import math
from queue import Queue
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from PIL import Image
from io import BytesIO
from pathlib import Path
from cryptography.fernet import Fernet
import numpy as np

# 确保 VLM_EXP_DEBUG=0 以执行 SSH 刷新网页状态
if os.environ.get('VLM_EXP_DEBUG') is None:
    os.environ['VLM_EXP_DEBUG'] = '0'

# 确保 WEBARENA_AUTH_PATH 环境变量存在
if os.environ.get('WEBARENA_AUTH_PATH') is None:
    os.environ['WEBARENA_AUTH_PATH'] = './log'

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from loguru import logger

# 导入 demo 下的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加 Agent-R1/eval 到 path 以便能找到 demo

# Modified imports for real implementation structure
from opagent.refresh import ssh_connect_and_refresh_gitlab
from opagent.task_scheduler import TaskScheduler
from opagent.browser_env.refresh_web import ssh_connect_and_refreshweb
from opagent.browser_env.async_envs import BrowserActor, get_ws_endpoint_list
from opagent.evaluation_harness import evaluator_router
from sub_shopping_prompt import shopping_navigation_prompt

# =============================================================================
# 模型配置
# =============================================================================

# MatrixLLM API Keys Config
FIXED_SEED = 42

# API Keys for MatrixLLM Pool (用于 Gemini 和 Qwen)

API_KEYS_MATRIX_MAPPED = {
    "qwen": [
    ''
    # You can add more 'qwen' keys here in the future
    ],
    "gemini": [
    ''
    # You can add more 'gemini' keys here in the future
    ],
    # The 'default' group for any other models (like Claude).
    # It can also have multiple keys for rotation.
    "default": [
    '',
    '' # Example: default can try both
    ]
}

# Reflector / Planner / Summary 使用的模型配置 (大模型，用于推理)
REASONING_MODEL_CONFIG = {
    # "model": "Qwen3-VL-235B-A22B-Instruct",
    # "temperature": 0.0,
    # "base_url": "https://antchat.alipay.com/v1",
    # "api_key": "",
    "model": "gemini-3-pro-preview",  # 使用 Gemini 模型
    "temperature": 0.0,
}


# Grounder 使用的模型配置 (SFT 微调过的模型，专门用于坐标定位)
GROUNDER_MODEL_CONFIG = {
    "model": "Qwen2.5-VL-72B-Instruct-SFT",
    "temperature": 0.0,
    "base_url": "",
    "api_key": "",
}

# 最大执行步数
MAX_STEPS = 70
# 执行超时时间 (秒)
TIMEOUT = 120 * 60

# 站点端口映射
SITE_PORT_MAP = {
    "7770": "shopping",
    "7780": "shopping_admin",
    "9999": "reddit",
    "8023": "gitlab",
    "8888": "wikipedia",
    "3000": "map",
    "4399": "homepage",
}


# =============================================================================
# Prompt 模板
# =============================================================================

REFLECTION_PROMPT = """
# Your Role
You are part of a web automation agent with the following architecture:

    Planner generates an execution plan and instructions -> Grounder outputs specific actions and parameters -> Reflector reviews whether the action was successful -> Planner receives reflection suggestions to adjust the execution plan and instructions.

You are the **"Reflector"** in this system. Your responsibilities are:

    1. Review the input context, adhering strictly to the principle of *basing your analysis 100% on observed facts*.
    2. Observe and reflect on the agent's executed plan, path, and actions to determine if the task is complete.
    3. Record any information that satisfies the user's request (`user_query`).

# Domain Specific Expert Knowledge
{tips}

# Your Tasks

## 1. Verify Task Success
Review the collected notes (`marked_note`) and the current screenshot to determine if the user's request (`user_query`) is complete, following this process:
    
    Step 1: 1.1 Determine if the page has been scrolled to the bottom.
    If the page has not been fully scrolled, prioritize continuing to scroll to gather all information.
    
    Step 2: 1.2 Review the currently collected data.   
    Based on the collected information in `marked_note`, determine if it **fully satisfies** the `user_query`.
    
    Step 3: 1.3 Review the current screenshot and page code.
    Based on the current `screenshot` and `html_simplify`, determine:
    - What elements are present on the page.
    - If the information on the page is incomplete and it hasn't been scrolled to the bottom, continue scrolling to get complete information.
    - Whether the elements on the page can **fully satisfy** the `user_query`.
    
    Step 4: 1.4 Determine if the last action was completed.
    Based on the `recent_10_step_details`, determine if the last action was successfully executed.
    
    Step 5: 1.5 Check if all sub-tasks in the `todo-list` are completed.
    If the `todo-list` is not empty, check if all tasks are marked as complete (status is '✅').
    
    Only if `all` of the above conditions are met can you determine the task is complete. Provide your reasoning in the `thought` field (without including the step subheadings) and set `is_task_done` to `true`.

## 2. Task Termination Conditions
You must stop the task immediately under the following conditions. Set `is_task_done` to `true` and provide the reason in `thought`:

*   **Infinite Loop Trap**: Based on `current_path`, if the step count reaches 30, and there are 3 consecutive identical **non-special actions** with **no change on the page**.
    *   **Special Action**: `scroll`. If the page has not reached the bottom, the agent is allowed to continuously scroll until there is **no more change on the page**. This does not require correction.
*   **Hard Blockers**:
    *   If a login is required to proceed, review the `current_path` to see if an attempt has already been made to bypass it.
    *   If no bypass has been attempted, do not consider it a hard blocker and continue the task.
    *   If a bypass has been attempted and failed, consider it a hard blocker and stop the task.
    *   If a CAPTCHA or human verification challenge appears, immediately consider it a hard blocker and stop the task.

# Core Principles
*   **Incremental Collection**: The final answer is accumulated by gathering information step-by-step. Do not wait until all information is found to start recording. For example, if the task is to "find 5 candidates," you must record the information of each candidate as an intermediate result as soon as you visit their detail page.
*   **Focus on Relevance**: The information recorded in the `note` must directly serve the user's final goal (`user_query`). For example, for a candidate's resume, record name, experience, and skills; for a product, record name, price, and specifications.
*   **Action Persistence**: If the agent needs to perform a text input action, the flow is "click input field -> type text". The Planner might send the same text input command to the Grounder multiple times until the input is complete. Note that this is not an ineffective repetition. 

## 3. Key Information Extraction
From the current `screenshot` and `html_simplify`, *find and extract new information relevant to the `user_query`.* Or, *incrementally extract key information relevant to the `user_query` that is present on the page but not yet recorded.* Your steps are:

    3.1 **Check for Duplicates**: Compare with the currently recorded information in `marked_note`.
    3.2 **Extract New Knowledge**: Identify key information that exists on the page but is not present in `marked_note`.
    3.3 **Take Action**: If any new information is found, set `mark_node` to `true` and record this new information in a structured format in the `note` field.
        - `note` structure:
            `<Summary Point>: <Detailed bullet-point description>`
        - `note` structure example:
            `John Doe's Resume Info: 1. Name: John Doe; 2. Age: 30; 3. Gender: Male`
            
# Task Inputs
**User Goal (user_query)**:

{user_query}

**Previous and Current Screenshots (screenshot)**

Among the image inputs, the first image is the screenshot from the previous step, and the second is the current screenshot. If the previous action was a click or input, the location of the action will be highlighted with a red border and fluorescent green fill in the previous screenshot.

**Page Code (html_simplify)**
{html_simplify}

**Recent 10-Step Execution Details (recent_10_step_details)**

{recent_10_step_details}

**Collected Notes (marked_note)**:

{marked_note}

**Execution Path (current_path)**:

{current_path}

**Current Todo-List Status (todolist_status)**:

{todolist_status}

**Task Tips (tips)**:

{tips}

# Your Output
You must complete the task requirements based on the inputs and provide your output, which includes:
- "thought" (Required List['str']): Describe the effect of the last action on the page, whether the tasks in `todolist_status` are complete, whether the current page information is relevant to the `user_query` and needs to be recorded, and how to better proceed with the task.
- "mark_node" (Required bool): Whether to mark this action node.
- "note" (Required str): A note for this action node.
- "is_task_done" (Required bool): Whether the current task is complete.
- "todo_list" (Required List[Dict]): A structured todo list to track task progress. Each item should have:
    - "id": A unique integer ID for the task (1, 2, 3, ...)
    - "description": A brief description of the sub-task
    - "status": One of "pending", "in_progress", "completed", or "failed"
  Based on the current progress, you should:
    - Mark completed sub-tasks as "completed"
    - Mark the current sub-task as "in_progress"
    - Mark failed attempts as "failed" (with reason in description)
    - Add new sub-tasks if needed to complete the user_query
  If `todolist_status` is empty or "None", create a new todo list based on the user_query.

You must generate your output in a JSON code block that adheres to the following format:
{{
  "thought": <A list of strings containing your detailed reasoning in bullet points>,
  "mark_node": <Whether to mark this action node>,
  "note": <A note for this action node>,
  "is_task_done": <Whether the current task is complete>,
  "todo_list": [
    {{"id": 1, "description": "...", "status": "completed"}},
    {{"id": 2, "description": "...", "status": "in_progress"}},
    {{"id": 3, "description": "...", "status": "pending"}}
  ]
}}

"""


PLANNER_PROMPT = """
# Your Role
You are a highly specialized Task Planner for a web AI agent. You must generate task instructions based 100% on the provided context, without making assumptions or guessing.

# Your Tasks

## Formulate a Specific Execution Plan
You need to follow the process below to generate a single, specific, and immediately executable action for the web page. This action will be output as an `instruction` to the downstream Grounder.

    `instruction` definition: A concise and clear action command for the Grounder.
    Please construct the instruction following this process:
    *   Briefly describe the target element's relative position (e.g., left-side menu, top-right corner, inside the central message box), appearance (e.g., green border, gray fill), and type (e.g., button, link, text). Avoid ambiguous positional descriptions like "the first xxx" or "the second xxx" that are hard for the Grounder to interpret.
    *   If a button or element contains text, include the text in the description, such as "'xxx Scenic Area' button".
    *   Briefly describe the action to be performed.
    *   Include any important parameters (e.g., text to be filled in, usernames, passwords from the user request).

1.  Review the current `todo_list_status`. If it's not empty, strictly follow the order and focus on the first unfinished item. Analyze how to complete it. If the list is empty, skip this step.

2.  Review the `reflection_signal` from the reflection module. If the signal is reasonable, adopt it. If it suggests an action on an element that does not exist on the page, state your reason for rejecting it in `thought` and do not adopt the instruction.

3.  Review the current `screenshot` to find key elements needed for the action (e.g., link text, input field location).

4.  Check for any valid `tips`. If present, treat them as **expert advice** or a **shortcut** and give them the highest priority. You must explicitly explain in your `thought` how you understand and are adopting the `tips`. If `tips` is empty or just says "tips", skip this step.

5.  Review the `recent_10_step_details`. Based on the context, provide the next action instruction in `instruction`.

6.  Describe the current page layout in your `thought`.

7.  Based on your reasoning in `thought`, provide the next action using `action_type` and `action_attributes`. The actions you can output are:
    *   **scroll**:
        *   Description: Scroll a **specific local section** or the **entire global page** to find information. You need to describe the features of the area you want to scroll to get its coordinates in the instruction.
        *   Output Format:
            *   `action_type`: 'scroll'
            *   `action_attributes`: `{{'direction': <'up'|'down'|'left'|'right'>}}`
            *   `instruction`: Based on the current page layout, describe the features of the area to be scrolled to get its coordinates. **Do not include action verbs like 'scroll'**. For example: "the center of the product recommendation list", "the center of the related articles module on the right side of the page".
    *   **go_back**:
        *   Description: Return to the previous page in the current tab. If it's the first page of the tab, it will go to the previous tab.
        *   Output Format:
            *   `action_type`: 'go_back'
            *   `action_attributes`: null
            *   `instruction`: No instruction needed, output null.
    *   **click**:
        *   Description: Click on an element on the page.
        *   Output Format:
            *   `action_type`: 'click'
            *   `action_attributes`: null
            *   `instruction`: Provide a detailed description of the element to be clicked. E.g., "Click the search box", "Click the first product".
    *   **type**:
        *   Description: Type text into an input field.
        *   Output Format:
            *   `action_type`: 'type'
            *   `action_attributes`: `{{'content': <text_to_type>, 'press_enter': <boolean>}}` (defaults to `true` for search boxes, but should be `false` when filling out multi-field forms like flight or train ticket bookings).
            *   `instruction`: A description of the input element. E.g., 'Type "..." into the search box at the top of the page'.
    *   **goto**:
        *   Description: Navigate to a specified URL.
        *   Output Format:
            *   `action_type`: 'goto'
            *   `action_attributes`: `{{'url': <url_to_go_to>}}`
            *   `instruction`: null
    *   **press**:
        *   Description: Press a key, like the Enter key.
        *   Output Format:
            *   `action_type`: 'press'
            *   `action_attributes`: `{{'key': 'Enter'}}`. **Note: The key name must be capitalized.**
            *   `instruction`: null
    *   **hover**:
        *   Description: Move the mouse cursor over an element to make it hover.
        *   Output Format:
            *   `action_type`: 'hover'
            *   `action_attributes`: null
            *   `instruction`: Provide a detailed description of the element to hover over. E.g., "Hover over the first product image", "Move the mouse over the search box".
    *   **select_option**:
        *   Description: Use this when you need to select a specific option from a dropdown list (typically a `<select>` tag).
        *   Output Format:
            *   `action_type`: 'select_option'
            *   `action_attributes`: `{{'option': <'text_value_of_the_option_to_select'>}}`. E.g.: `{{'option': 'California'}}`
            *   `instruction`: Describe the **dropdown menu itself** that you want to operate on, and explicitly instruct the Grounder to **click** it. E.g.: "Click the dropdown menu for selecting 'State'", "Click the 'Sort by' dropdown". **Note: The instruction must include the verb 'click' so the Grounder can correctly understand and return the coordinates.**
    *   **right_click**:
        *   Description: Perform a right-click (context menu) on an element. Useful for accessing context menus, such as "Show address" on a map.
        *   Output Format:
            *   `action_type`: 'right_click'
            *   `action_attributes`: null
            *   `instruction`: Describe the element to right-click on. E.g., "Right-click on the red marker on the map", "Right-click on the location pin".
    *   **zoom_in**:
        *   Description: Zoom in on the page or a specific area (e.g., a map). This simulates pressing Ctrl/Cmd + '+' or clicking a zoom-in button.
        *   Output Format:
            *   `action_type`: 'zoom_in'
            *   `action_attributes`: `{{'level': <number_of_zoom_steps, default 1>}}` (optional)
            *   `instruction`: Describe the area to zoom in on if applicable. E.g., "Zoom in on the map", "Zoom in on the center of the page".
    *   **zoom_out**:
        *   Description: Zoom out on the page or a specific area (e.g., a map). This simulates pressing Ctrl/Cmd + '-' or clicking a zoom-out button.
        *   Output Format:
            *   `action_type`: 'zoom_out'
            *   `action_attributes`: `{{'level': <number_of_zoom_steps, default 1>}}` (optional)
            *   `instruction`: Describe the area to zoom out on if applicable. E.g., "Zoom out on the map", "Zoom out to see more of the page".

# Core Task Principles

*   The instructions you give to the Grounder must be within its capabilities (click, type, hover, etc.). Do not command actions outside its scope.
    *   Correct example: 'Type "Software Engineer" into the search box.'
    *   Incorrect example: 'Enter "https://xxx" into the browser's address bar.'
*   The output action must be a single, atomic operation for the next step. It cannot contain multiple actions.
*   For information retrieval tasks, prioritize using the on-page search box.
*   If you need to select a value from a dropdown menu, use the `select_option` action.
*   If the page content does not change after typing text, try clicking a nearby button like 'Search', 'Query', or a magnifying glass icon.
*   If the screenshot is blank after an action, the page might not have fully loaded. You can wait for one turn before proceeding.
*   When filling out forms, the Grounder will automatically clear any existing placeholder text; no instruction is needed for this. Be aware of faint, grayed-out text inside input fields (e.g., 'Password', 'Username'). These should be treated as empty input fields.
*   If dropdown options appear after clicking an input field, first check if any option contains the target keyword. If so, use `click` to select it.
*   If you believe you are on the correct page but the answer is not visible, use the scroll action to move up and down the page to search for it.

# Task Input Information
**User Goal (user_query)**:
{user_query}

**Task Tips (tips)**:
{tips}

**Screenshot (screenshot)**:
[Image provided in the input]

**Current To-Do List Status (todo_list_status)**:
{todo_list_status}

**Reflection Signal (reflection_signal)**:
{reflection_signal}

**Collected Notes (marked_note)**:
{marked_note}

**Recent 3-Step Execution Details (recent_10_step_details)**:
{recent_10_step_details}

# Your Output
Guided by the core principles, provide your reasoning. Your output must be a JSON-parsable string without any comments or extra characters:

{{
    "thought": "<Your thought process on how to complete the to-do item, how to use useful information on the page, whether to adopt the reflection signal and why, and the reasoning for your chosen action>",
    "instruction": "<The concise and clear text instruction for the Grounder>",
    "action_type": "<The action object you are outputting. Output null if no suitable action is found>",
    "action_attributes": "<The parameters for your action. Defaults to null>"
}}
"""



GROUNDER_PROMPT = """You are an excellent web agent. 
Now, you are given a user query along with the current webpage (including screenshot and other information). 
You need to call the provided webpage functions multiple times to complete the user's request. 
Considering the current state of the webpage and the user's request, please give a required webpage function and its corresponding parameters for each round of conversation, and output them strictly according to the following format.
If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<tool_call>
...
</tool_call>

Output format for answer:
<answer>
...
</answer>
<image>
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction:{instruction}



"""

SUMMARY_PROMPT = """
Based on the following task execution history and sequence of page screenshots, summarize the final result of the task:

**User Goal**:
{user_query}

**Task Tips**:
{tips}

**Execution History**:
{execution_history}

**Collected Notes (marked_note)**:
{marked_note}

**Screenshot Sequence** (A total of {screenshot_count} images, ordered chronologically):
The image inputs contain the screenshot sequence from [Image 1] to [Image {screenshot_count}].

Please carefully analyze the content changes across the screenshot sequence and the collected notes to understand the task's execution process and its final state.
Summarize the task's result concisely to answer the user's question.

Output Format:
{{
  "answer": "<The task result/answer>",
  "success": <boolean, true if the task was successfully completed, false otherwise>
}}

"""


# =============================================================================
# AFTS 图片上传工具（用于 Grounder 模型，必须使用图片 URL）
# =============================================================================

class AFTSTool:
    """AFTS 文件上传工具，用于将截图上传到 AFTS 获取 URL"""
    appid: str = "apwallet"
    biz_key: str = "oagent"
    biz_secret: str = Fernet(b"***REMOVED-FERNET-KEY***=").decrypt(
        b"***REMOVED-FERNET-CIPHERTEXT***"
    ).decode('utf-8')
    upload_endpoint_source: str = "mass.alipay.com"
    authority_endpoint: str = "mmtcapi.alipay.com"
    http_schema: str = 'https'

    def _get_op_token(self):
        """获取 op token (同步版本)"""
        time_stamp = str(int(time.time() * 1000))
        authority_url = self.http_schema + "://" + self.authority_endpoint + "/token/1.0/op"
        url_params = {"timestamp": time_stamp, "bizKey": self.biz_key, "appId": self.appid}

        md5_handle = hashlib.md5()
        md5_handle.update((self.appid + self.biz_key + time_stamp + self.biz_secret).encode('utf-8'))
        url_params["sign"] = md5_handle.hexdigest()

        response = requests.get(authority_url, params=url_params)
        if response.status_code != 200:
            logger.error(f"get_op_token Error: http status code != 200, message: {response.text}")
            return None
        else:
            res_json = response.json()
            if res_json['code'] != 0:
                logger.error(f"get_op_token Error: server response code != 0, code: {res_json['code']}")
                return None
            else:
                return res_json["data"]["token"]

    def _get_mass_token(self):
        """获取 mass token (同步版本)"""
        authority_url = self.http_schema + "://" + self.authority_endpoint + "/token/1.0/mass"
        url_params = {"appId": self.appid, "bizKey": self.biz_key, "opToken": self._get_op_token(), "massType": '1'}

        response = requests.get(authority_url, params=url_params)
        if response.status_code != 200:
            logger.error(f"get_mass_token Error: http status code != 200, message: {response.text}")
            return None
        else:
            res_json = response.json()
            if res_json['code'] != 0:
                logger.error(f"get_mass_token Error: server response code != 0, code: {res_json['code']}")
                return None
            else:
                return res_json["data"]

    def upload_file(self, file_data: bytes, file_name: str, setpublic: bool = True) -> Optional[str]:
        """上传文件到 AFTS，返回文件 ID (同步版本)"""
        mass_token = self._get_mass_token()
        if not mass_token:
            return None
            
        upload_url = "https://" + self.upload_endpoint_source + "/file/auth/upload"
        url_params = {"bz": self.biz_key, "public": str(setpublic).lower(), "mt": mass_token}
        form_file = {"file": (file_name, file_data, "application/octet-stream")}

        response = requests.post(upload_url, params=url_params, files=form_file)
        if response.status_code != 200:
            logger.error(f"upload_file Error: http status code != 200, message: {response.text}")
            return None
        else:
            res_json = response.json()
            if res_json['code'] != 0:
                logger.error(f"upload_file Error: server response code != 0, code: {res_json['code']}")
                return None
            else:
                _file_id = res_json["data"]["id"]
                logger.debug(f"成功上传文件: {file_name}, file_id: {_file_id}")
                return _file_id

    def get_file_url(self, file_id: str) -> str:
        """获取文件下载 URL"""
        download_url = "https://" + self.upload_endpoint_source + "/afts/file/" + file_id + "?bizType=" + self.biz_key
        return download_url

    def upload_base64_image(self, base64_data: str, file_name: str = "screenshot.png") -> Optional[str]:
        """将 base64 图片上传到 AFTS 并返回 URL (同步版本)"""
        try:
            # 解码 base64
            image_data = base64.b64decode(base64_data)
            # 上传文件
            file_id = self.upload_file(image_data, file_name)
            if file_id:
                # 获取 URL
                return self.get_file_url(file_id)
            return None
        except Exception as e:
            logger.error(f"上传 base64 图片失败: {e}")
            return None


# 全局 AFTS 工具实例
_afts_tool = None

def get_afts_tool() -> AFTSTool:
    """获取全局 AFTS 工具实例"""
    global _afts_tool
    if _afts_tool is None:
        _afts_tool = AFTSTool()
    return _afts_tool


# =============================================================================
# 坐标缩放相关函数 
# =============================================================================

MAX_RATIO = 200  # 图像最大宽高比

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 784, max_pixels: int = 3211264) -> Tuple[int, int]:
    """
    制定模型输入的图片的压缩比例
    和 control_agent 完全一致
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
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

def get_image_size_from_url(image_url: str) -> Dict[str, Any]:
    """获取线上图片的尺寸"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as img:
            width, height = img.size
        return {'width': width, 'height': height}
    except Exception as e:
        logger.warning(f"获取图片尺寸失败: {e}")
        return {'width': None, 'height': None}

def get_image_size_from_base64(base64_str: str) -> Dict[str, Any]:
    """获取 base64 图片的尺寸"""
    try:
        # 移除可能的 data URL 前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        image_data = base64.b64decode(base64_str)
        with Image.open(BytesIO(image_data)) as img:
            width, height = img.size
        return {'width': width, 'height': height}
    except Exception as e:
        logger.warning(f"获取 base64 图片尺寸失败: {e}")
        return {'width': None, 'height': None}

def image_size_extract(image_url: str) -> Tuple[Optional[int], Optional[int], float, float]:
    """
    处理图片链接，获取到图片被压缩后的尺寸和还原指数
    和 control_agent 完全一致
    """
    image_size_dict = get_image_size_from_url(image_url)
    width = image_size_dict.get('width', None)
    height = image_size_dict.get('height', None)
    if (width is not None) and (height is not None):
        resized_height, resized_width = smart_resize(height, width)
        # 获取尺寸还原指数
        height_revert = height / resized_height
        width_revert = width / resized_width
        logger.info(f"图片尺寸获取成功，原始: {width}x{height}, 压缩后: {resized_width}x{resized_height}, 还原指数: h={height_revert:.2f}, w={width_revert:.2f}")
        return resized_height, resized_width, height_revert, width_revert
    logger.warning(f"图片尺寸获取失败, 还原指数指定为1")
    return None, None, 1.0, 1.0

def revert_coords(coords: List[int], image_url: str) -> List[int]:
    """
    将模型输出的坐标（基于压缩图像）还原到原始图像坐标
    和 control_agent/qwen_72b_multi_image.py 的逻辑一致
    """
    if not coords or len(coords) < 2:
        return coords
    
    resized_height, resized_width, height_revert, width_revert = image_size_extract(image_url)
    
    coordinate_x, coordinate_y = coords[0], coords[1]
    r_coordinate_x = round(int(coordinate_x) * width_revert)
    r_coordinate_y = round(int(coordinate_y) * height_revert)
    
    logger.info(f"坐标还原: 原始坐标 ({coordinate_x}, {coordinate_y}) -> 还原坐标 ({r_coordinate_x}, {r_coordinate_y})")
    return [r_coordinate_x, r_coordinate_y]


# =============================================================================
# Grounder 调用 
# =============================================================================

# Grounder 专用 OpenAI 客户端 (和 control_agent 完全一致)
_grounder_client = None

def get_grounder_client():
    """获取 Grounder OpenAI 客户端"""
    global _grounder_client
    if _grounder_client is None:
        from openai import OpenAI
        _grounder_client = OpenAI(
            api_key='b12bf879-03a1-8942-a6f9-f34edef3a32f',
            base_url="https://codebot.alipay.com/v1"
        )
    return _grounder_client


def generate_grounder_message(prompt: str, image_url_list: List[str]) -> List[Dict]:
    """
    生成 Grounder 消息格式 (和 control_agent 完全一致)
    """
    content = []
    for i in range(len(image_url_list)):
        if i == 0:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url_list[i]},
                "min_pixels": 784,
                "max_pixels": 784 * 4096
            })
        else:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url_list[i]},
                "min_pixels": 784,
                "max_pixels": 784 * 384
            })
    # 添加 prompt 部分
    content.append({"type": "text", "text": prompt})

    image_message = [
        {
            "role": "system",
            "content": "你是一个高级网页代理"
        },
        {
            "role": "user",
            "content": content
        }
    ]
    return image_message


def parse_grounder_tool_call(model_response: str) -> Optional[Dict]:
    """
    解析 Grounder 返回的 tool_call (和 control_agent 完全一致)
    """
    try:
        tool_call = model_response.split("<tool_call>")[1].split("</tool_call>")[0]
        tool_call = json.loads(tool_call)
        return tool_call
    except Exception as e:
        logger.warning(f"解析 Grounder tool_call 失败：{e}")
        return None


def call_grounder_api(image_url_list: List[str], prompt: str) -> Optional[Dict]:
    """
    调用 Grounder API (和 control_agent/qwen_72b_multi_image.py 的 ground_api 完全一致)
    
    注意：这是同步函数，使用同步的 OpenAI SDK
    """
    client = get_grounder_client()
    
    # 生成消息
    messages = generate_grounder_message(prompt=prompt, image_url_list=image_url_list)
    
    logger.info(f"[Grounder] Calling API, images={len(image_url_list)}, prompt={prompt[:100]}...")
    
    try:
        #messages[1]['content'][1]['text'] = '<image>' + messages[1]['content'][1]['text']
        chat_response = client.chat.completions.create(
            model="Qwen2.5-VL-72B-Instruct-SFT",
            messages=messages,
            stream=False
        )
        
        logger.debug(f"[Grounder] Raw response: {chat_response}")
        
        if not chat_response.choices or len(chat_response.choices) == 0:
            logger.error(f"[Grounder] Empty choices in response")
            return None
        
        model_response = chat_response.choices[0].message.content
        logger.info(f"[Grounder] Model response: {model_response[:200] if model_response else 'None'}...")
        
        if not model_response:
            logger.error(f"[Grounder] Empty model response")
            return None
        
        # 解析 tool_call
        tool_call_action = parse_grounder_tool_call(model_response)
        logger.info(f"[Grounder] Parsed action: {tool_call_action}")
        
        return tool_call_action
        
    except Exception as e:
        logger.error(f"[Grounder] API call failed: {e}")
        traceback.print_exc()
        return None


def upload_and_call_grounder(base64_image: str, prompt: str) -> Optional[Dict]:
    """
    上传图片并调用 Grounder
    
    1. 将 base64 图片上传到 AFTS 获取 URL
    2. 调用 Grounder API
    """
    afts_tool = get_afts_tool()
    
    # 上传图片
    image_url = afts_tool.upload_base64_image(base64_image, f"screenshot_{int(time.time())}.png")
    if not image_url:
        logger.error("[Grounder] Failed to upload image to AFTS")
        return None
    
    logger.info(f"[Grounder] Image uploaded: {image_url[:80]}...")
    
    # 调用 Grounder API
    return call_grounder_api([image_url], prompt)


# =============================================================================
# 模型调用封装
# =============================================================================

class LocalModelCaller:
    """本地模型调用器，封装对 VLM 模型的调用"""
    
    def __init__(self, config: Dict[str, Any], is_grounder: bool = False):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:8000/v1')
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'Qwen2.5-VL-72B-Instruct')
        self.temperature = config.get('temperature', 0.0)
        self.is_grounder = is_grounder
        
        # Grounder 使用 codebot API，需要用 OpenAI SDK
        self.use_openai_sdk = 'codebot.alipay.com' in self.base_url
        # MatrixLLM / Gemini detection
        self.use_matrix = 'matrixllm-pool' in self.base_url or 'gemini' in self.model.lower()
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析模型响应"""
        import json_repair
        
        # 1. 尝试解析 <tool_call> 格式 (Grounder SFT 模型输出)
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', response_text, re.DOTALL)
        if tool_call_match:
            try:
                tool_call_str = tool_call_match.group(1).strip()
                return json_repair.loads(tool_call_str)
            except Exception as e:
                logger.warning(f"Failed to parse tool_call: {e}")
        
        # 2. 尝试解析 <answer> 格式
        answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if answer_match:
            return {"answer": answer_match.group(1).strip(), "is_answer": True}
        
        # 3. 尝试解析 JSON 代码块
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                return json_repair.loads(json_str)
            except Exception as e:
                logger.warning(f"Failed to parse JSON: {e}")
        
        # 4. 返回原始响应
        return {"raw_response": response_text}
    
    async def call(self, prompt: str, image_list: List[str] = None, 
                   max_retries: int = 10) -> Dict[str, Any]:
        """调用 VLM 模型"""
        if self.use_matrix:
            # MatrixLLM Pool (Gemini / Qwen)
            return await self._call_matrix(prompt, image_list, max_retries)
        elif self.use_openai_sdk:
            # codebot API 使用 OpenAI SDK (Grounder)
            return await self._call_openai_sdk(prompt, image_list, max_retries)
        else:
            # antchat API 使用 HTTP (Reflector/Planner)
            return await self._call_http(prompt, image_list, max_retries)
    
    async def _call_matrix(self, prompt: str, image_list: List[str] = None, max_retries: int = 10) -> Dict[str, Any]:
        """使用 MatrixLLM Pool 调用 (支持 Key Rotation)"""
        try:
            from openai import OpenAI, APIError
        except ImportError:
            logger.error("OpenAI SDK not installed.")
            return {}

        # 环境变量清理
        if 'HTTP_PROXY' in os.environ: del os.environ['HTTP_PROXY']
        if 'HTTPS_PROXY' in os.environ: del os.environ['HTTPS_PROXY']
        
        base_url = 'http://matrixllm-pool.global.alipay.com/v1'
        retry_delay = 10

        # --- KEY LIST SELECTION LOGIC ---
        selected_key_list = None
        model_name_lower = self.model.lower()
        
        if "qwen" in model_name_lower:
            selected_key_list = API_KEYS_MATRIX_MAPPED.get("qwen")
        elif "gemini" in model_name_lower:
            selected_key_list = API_KEYS_MATRIX_MAPPED.get("gemini")
        
        if not selected_key_list:
            logger.warning(f"No specific key list for model '{self.model}'. Using default key list.")
            selected_key_list = API_KEYS_MATRIX_MAPPED.get("default")

        if not selected_key_list or not isinstance(selected_key_list, list) or len(selected_key_list) == 0:
            logger.error(f"FATAL: No valid API key list found for model '{self.model}'.")
            return {}
        # --- END OF KEY LIST SELECTION LOGIC ---

        # 准备消息 (直接使用 base64，不上传 AFTS)
        content = []

        # 1. 先添加图片 (resize到1024后使用 base64 格式)
        if image_list:
            for i, img in enumerate(image_list):
                if img and img != '' and img != 'None':
                    # 处理图片格式：如果是 URL 则跳过
                    if img.startswith('http'):
                        logger.warning(f"[Matrix] 图片 {i} 是 URL 格式，跳过")
                        continue
                    
                    # Resize 到 1024 并编码为 base64 data URI
                    img_data_uri = resize_image_to_base64(img, max_size=1024)
                    
                    if img_data_uri:
                        logger.info(f"[Matrix] 图片 {i} 已resize到1024并使用base64格式 (长度: {len(img_data_uri)})")
                        # 构造 Image Message，直接使用 base64 data URI
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img_data_uri},
                        })

        # 2. 再添加文本
        content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "system", "content": "你是一个高级网页代理"},
            {"role": "user", "content": content}
        ]

        attempts = 0
        while attempts < max_retries:
            current_key = selected_key_list[attempts % len(selected_key_list)]
            
            client = OpenAI(api_key=current_key, base_url=base_url, max_retries=0, timeout=10 * 60.0)
            
            try:
                logger.info(f"Attempt {attempts + 1}/{max_retries} for model '{self.model}' with key: {current_key[:8]}...")
                
                # Call create (sync, blocking)
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    seed=FIXED_SEED
                )
                
                if not chat_response.choices:
                    raise Exception("Empty choices")
                    
                response_text = chat_response.choices[0].message.content
                logger.debug(f"[Matrix] Response: {response_text[:200]}")
                return self._parse_response(response_text)
                
            except APIError as e:
                attempts += 1
                logger.warning(f"API Request Failed: {e}")
                if attempts < max_retries: 
                    await asyncio.sleep(retry_delay)
            except Exception as e:
                attempts += 1
                logger.error(f"An unexpected error occurred: {e}")
                if attempts < max_retries: 
                    await asyncio.sleep(retry_delay)
        
        return {}
    
    async def _call_openai_sdk(self, prompt: str, image_list: List[str] = None,
                                max_retries: int = 3) -> Dict[str, Any]:
        """使用 OpenAI SDK 调用 (用于 codebot.alipay.com Grounder)
        
        注意：codebot.alipay.com 不支持 base64 图片，必须使用图片 URL。
        因此需要先将 base64 图片上传到 AFTS 获取 URL。
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("OpenAI SDK not installed. Run: pip install openai")
            return {}
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 获取 AFTS 工具
        afts_tool = get_afts_tool()
        
        # 构建消息内容 - 图片在前，文本在后（和 control_agent 一致）
        content = []
        
        # 1. 先添加图片 - 必须使用 URL 格式（codebot 不支持 base64）
        if image_list:
            for i, img in enumerate(image_list):
                if img and img != '' and img != 'None':
                    # 确定图片 URL
                    if img.startswith('http'):
                        # 已经是 URL
                        img_url = img
                    elif img.startswith('data:'):
                        # data URI 格式，提取 base64 部分并上传
                        base64_data = img.split(',', 1)[1] if ',' in img else img
                        uploaded_url = await afts_tool.upload_base64_image(
                            base64_data, 
                            f"screenshot_{i}_{int(time.time())}.png"
                        )
                        if uploaded_url:
                            img_url = uploaded_url
                            logger.debug(f"[Grounder] 图片 {i} 上传成功: {img_url[:80]}...")
                        else:
                            logger.warning(f"[Grounder] 图片 {i} 上传失败，跳过")
                            continue
                    else:
                        # 纯 base64 格式，上传到 AFTS
                        uploaded_url = await afts_tool.upload_base64_image(
                            img, 
                            f"screenshot_{i}_{int(time.time())}.png"
                        )
                        if uploaded_url:
                            img_url = uploaded_url
                            logger.debug(f"[Grounder] 图片 {i} 上传成功: {img_url[:80]}...")
                        else:
                            logger.warning(f"[Grounder] 图片 {i} 上传失败，跳过")
                            continue
                    
                    # 第一张图片用更高分辨率
                    if i == 0:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img_url},
                            "min_pixels": 784,
                            "max_pixels": 784 * 4096
                        })
                    else:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img_url},
                            "min_pixels": 784,
                            "max_pixels": 784 * 384
                        })
        
        # 2. 再添加文本
        content.append({"type": "text", "text": prompt})
        
        # 构建消息 - 包含 system 消息（和 control_agent 一致）
        messages = [
            {"role": "system", "content": "你是一个高级网页代理"},
            {"role": "user", "content": content}
        ]
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[Grounder] Calling OpenAI SDK, model={self.model}, images={len(image_list) if image_list else 0}")
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False
                )
                
                # 打印完整响应用于调试
                logger.info(f"[Grounder] Response object: {response}")
                
                # 检查响应
                if not response or not response.choices or len(response.choices) == 0:
                    logger.warning(f"OpenAI SDK call attempt {attempt + 1}: Empty choices, response={response}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    continue
                
                response_text = response.choices[0].message.content
                if response_text is None:
                    logger.warning(f"OpenAI SDK call attempt {attempt + 1}: Response content is None")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    continue
                
                logger.debug(f"[OpenAI SDK] Raw response: {response_text[:200]}")
                return self._parse_response(response_text)
                
            except Exception as e:
                logger.error(f"OpenAI SDK call attempt {attempt + 1} failed: {e}")
                traceback.print_exc()
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        return {}
    
    async def _call_http(self, prompt: str, image_list: List[str] = None,
                          max_retries: int = 3) -> Dict[str, Any]:
        """使用 HTTP API 调用"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 构建消息内容 - 文本在前，图片在后（和 control_agent 一致）
        content = [{"type": "text", "text": prompt}]
        if image_list:
            for img in image_list:
                if img and img != '' and img != 'None':
                    # 判断是 URL 还是 base64
                    if img.startswith('http'):
                        content.append({"type": "image_url", "image_url": {"url": img}})
                    elif img.startswith('data:'):
                        content.append({"type": "image_url", "image_url": {"url": img}})
                    else:
                        # 假设是 base64
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature
        }
        
        url = f"{self.base_url}/chat/completions"
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Model API error: {response.status} - {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2)
                            continue
                        
                        result = await response.json()
                        
                        # 检查响应格式
                        if not result or 'choices' not in result or len(result['choices']) == 0:
                            logger.warning(f"HTTP call attempt {attempt + 1}: Empty response")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2)
                            continue
                        
                        response_text = result['choices'][0]['message']['content']
                        if response_text is None:
                            logger.warning(f"HTTP call attempt {attempt + 1}: Response content is None")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2)
                            continue
                            
                        return self._parse_response(response_text)
                            
            except Exception as e:
                logger.error(f"HTTP call attempt {attempt + 1} failed: {e}")
                traceback.print_exc()
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    
        return {}


# =============================================================================
# 步骤信息
# =============================================================================

@dataclass
class StepInfo:
    """单步执行信息"""
    step_id: int
    action: Dict[str, Any]
    screenshot_path: str
    screenshot_base64: str = ""
    page_url: str = ""
    error_msg: str = ""
    reflection: Dict[str, Any] = field(default_factory=dict)
    planner_output: Dict[str, Any] = field(default_factory=dict)
    grounder_output: Dict[str, Any] = field(default_factory=dict)
    note: str = ""  # Reflector 收集的备注信息


# =============================================================================
# 浏览器操作 
# =============================================================================

def get_coords_from_params(params: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """从params中提取坐标"""
    if "coords" in params:
        coords = params["coords"]
        if isinstance(coords, list) and len(coords) >= 2:
            return coords[0], coords[1]
    if "x" in params and "y" in params:
        return params["x"], params["y"]
    if "coordinate_x" in params and "coordinate_y" in params:
        return params["coordinate_x"], params["coordinate_y"]
    return None, None




async def execute_browser_action(page: Page, action_type: str, params: Dict[str, Any]) -> str:
    """执行浏览器动作"""
    error_msg = ""
    x, y = get_coords_from_params(params)
    
    try:
        if action_type == "click":
            if x is not None and y is not None:
                await page.mouse.click(x, y)
            elif "selector" in params:
                await page.click(params["selector"], timeout=5000)
            else:
                logger.warning(f"Click action missing coords: {params}")
                
        elif action_type == "double_click":
            if x is not None and y is not None:
                await page.mouse.dblclick(x, y)
                
        elif action_type in ["input", "type"]:
            content = params.get("text", params.get("content", ""))
            press_enter = params.get("press_enter", params.get("press_enter_after", False))
            clear_before = params.get("clear_before_input", True)
            
            if x is not None and y is not None:
                await page.mouse.click(x, y)
                await asyncio.sleep(0.2)
            
            if clear_before:
                await page.keyboard.press("Control+a")
                await asyncio.sleep(0.1)
                await page.keyboard.press("Backspace")
                await asyncio.sleep(0.1)
            
            await page.keyboard.type(content, delay=30)
            
            if press_enter:
                await asyncio.sleep(0.1)
                await page.keyboard.press("Enter")
                
        elif action_type == "scroll":
            direction = params.get("direction", "down")
            distance = params.get("pixel_amount", params.get("distance", 300))
            
            if direction == "down":
                await page.evaluate(f"window.scrollBy(0, {distance})")
            elif direction == "up":
                await page.evaluate(f"window.scrollBy(0, -{distance})")
            elif direction == "right":
                await page.evaluate(f"window.scrollBy({distance}, 0)")
            elif direction == "left":
                await page.evaluate(f"window.scrollBy(-{distance}, 0)")
                
        elif action_type in ["goto", "go_to_url"]:
            url = params.get("url", "")
            if url:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
        elif action_type == "press":
            key = params.get("key", "Enter")
            if x is not None and y is not None:
                await page.mouse.click(x, y)
            await page.keyboard.press(key)
            
        elif action_type == "hover":
            if x is not None and y is not None:
                await page.mouse.move(x, y)
                
        elif action_type == "go_back":
            try:
                await page.go_back(wait_until='domcontentloaded', timeout=30000)
            except Exception as e:
                logger.warning(f"go_back failed or no history: {e}")
            
        elif action_type == "go_forward":
            try:
                await page.go_forward(wait_until='domcontentloaded', timeout=30000)
            except Exception as e:
                logger.warning(f"go_forward failed or no forward history: {e}")
            
        elif action_type == "right_click":
            if x is not None and y is not None:
                await page.mouse.click(x, y, button="right")
            else:
                logger.warning(f"Right click action missing coords: {params}")
                
        elif action_type == "zoom_in":
            level = params.get("level", 1)
            # 如果有坐标，先点击该位置然后缩放
            if x is not None and y is not None:
                await page.mouse.click(x, y)
                await asyncio.sleep(0.2)
            # 使用键盘快捷键缩放
            for _ in range(level):
                await page.keyboard.press("Control+Equal")  # Ctrl + '+'
                await asyncio.sleep(0.3)
                
        elif action_type == "zoom_out":
            level = params.get("level", 1)
            # 如果有坐标，先点击该位置然后缩放
            if x is not None and y is not None:
                await page.mouse.click(x, y)
                await asyncio.sleep(0.2)
            # 使用键盘快捷键缩放
            for _ in range(level):
                await page.keyboard.press("Control+Minus")  # Ctrl + '-'
                await asyncio.sleep(0.3)

######################################
        elif action_type == "select_option":
            option_text = params.get("option")
            
            if x is not None and y is not None and option_text:
                try:
                    # 步骤 1: 点击坐标，强制打开下拉菜单
                    # 这确保了即使坐标指向的是一个包裹 <select> 的 <div>，也能触发下拉。
                    await page.mouse.click(x, y)
                    # 给予一点时间让系统UI渲染出来
                    await asyncio.sleep(0.3)
                    
                    # 步骤 2: 使用 page.select_option 选择
                    # 现在下拉菜单已经打开，Playwright可以更好地识别选项。
                    # 我们需要找到被点击的 <select> 元素。
                    # Playwright 没有直接从坐标获取元素的方法，但我们可以用JS找到它，然后传递给 locator。
                    
                    # 用JS找到精确的元素
                    target_selector = await page.evaluate(f'''() => {{
                        const el = document.elementFromPoint({x}, {y});
                        if (!el) return null;
                        
                        // 向上查找，直到找到 <select> 或 body
                        let target = el;
                        for (let i=0; i<5; i++) {{ // 最多向上找5层
                            if (target.tagName === 'SELECT') break;
                            if (!target.parentElement || target.parentElement.tagName === 'BODY') break;
                            target = target.parentElement;
                        }}
                        
                        // 如果没找到 <select>，就用原始点击的元素
                        if (target.tagName !== 'SELECT') {{
                            target = el; 
                        }}
                        
                        // 生成一个唯一的CSS选择器
                        if (target.id) return `#${{target.id}}`;
                        if (target.name) return `[name="${{target.name}}"]`;
                        // 备用选择器，可能不稳定但聊胜于无
                        return `*:nth-of-type(1)`; 
                    }}''')
                    
                    if target_selector:
                        # 使用生成的选择器和标签文本进行选择
                        await page.select_option(target_selector, label=option_text, timeout=3000)
                    else:
                        raise Exception("Could not generate a selector for the element at the coordinates.")

                except Exception as e:
                    # 如果上述方法失败，回到最原始但可能有效的JS方法
                    logger.warning(f"Playwright's select_option failed: {e}. Falling back to JS execution.")
                    try:
                        js_code = f'''
                        (async (optionText) => {{
                            let el = document.elementFromPoint({x}, {y});
                            if (!el) return false;
                            
                            // 如果点击的不是 select，向上查找
                            if (el.tagName !== 'SELECT') {{
                                const parentSelect = el.closest('select');
                                if (parentSelect) el = parentSelect;
                            }}

                            if (el && el.tagName === 'SELECT') {{
                                for (const option of el.options) {{
                                    // 使用 includes 增加匹配的鲁棒性，应对 " 2022 " 这种情况
                                    if (option.text.trim().includes(optionText.trim())) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                }}
                            }}
                            return false;
                        }})("{option_text}")
                        '''
                        success = await page.evaluate(js_code)
                        if not success:
                            error_msg = f"JS fallback for select_option also failed for option: {option_text}"
                            logger.error(error_msg)
                            
                    except Exception as e2:
                        error_msg = f"JS fallback for select_option threw an error: {e2}"
                        logger.error(error_msg)
            else:
                error_msg = f"select_option action missing coords or option text. Params: {params}"
                logger.warning(error_msg)
##############################################

        elif action_type in ["stop", "finish", "done", "answer"]:
            pass  # 终止动作

        else:
            error_msg = f"Unknown action type: {action_type}"
            logger.warning(error_msg)


        # 等待页面稳定
        await asyncio.sleep(1)
        try:
            await page.wait_for_load_state('networkidle', timeout=10000)
        except:
            pass
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Action execution error: {e}")
    
    return error_msg


async def screenshot_to_base64(page: Page) -> str:
    """截取页面截图并转为 base64"""
    screenshot_bytes = await page.screenshot(full_page=False)
    return base64.b64encode(screenshot_bytes).decode('utf-8')



def resize_image_to_base64(image_base64: str, max_size: int = 1024) -> str:
    """
    将 base64 图像 resize 到指定最大尺寸并编码为 base64 data URI
    
    Args:
        image_base64: base64 字符串（可以是纯base64或data URI格式）
        max_size: 最大边长（保持宽高比）
    
    Returns:
        data URI 格式的 base64 字符串 (data:image/png;base64,...)
    """
    try:
        # 处理 data URI 格式：提取 base64 部分
        if image_base64.startswith('data:'):
            # 提取 base64 数据部分
            base64_data = image_base64.split(',', 1)[1]
        else:
            # 纯 base64 字符串
            base64_data = image_base64
        
        # 解码 base64 为图像
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes))
        
        # 转换 RGBA 到 RGB（如果需要）
        if image.mode == 'RGBA':
            # 创建白色背景
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # 使用alpha通道作为mask
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize：保持宽高比，最大边为 max_size
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 编码为 base64
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 返回 data URI 格式
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"resize_image_to_base64 失败: {e}")
        traceback.print_exc()
        # 如果处理失败，返回原始数据（如果已经是data URI格式）或添加前缀
        if image_base64.startswith('data:'):
            return image_base64
        else:
            return f"data:image/png;base64,{image_base64}"




async def save_screenshot(page: Page, save_path: str) -> str:
    """保存截图到文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    await page.screenshot(path=save_path, full_page=False)
    return save_path


# =============================================================================
# LocalWebAgent - 本地 WebAgent
# =============================================================================

class LocalWebAgent:
    """本地 WebAgent，直接调用 Planner、Reflector、Grounder"""
    
    def __init__(self, 
                 page: Page,
                 user_query: str,
                 output_dir: str,
                 reasoning_config: Dict[str, Any] = None,
                 grounder_config: Dict[str, Any] = None,
                 max_steps: int = MAX_STEPS,
                 timeout: int = TIMEOUT,
                 REPLACE_WITH_YOUR_HOST: str = None,
                 original_target_url: str = None):
        self.page = page
        self.user_query = user_query
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.timeout = timeout
        self.REPLACE_WITH_YOUR_HOST = REPLACE_WITH_YOUR_HOST or "REPLACE_WITH_YOUR_HOST"  # 用于 prompt/tips 动态替换
        self.original_target_url = original_target_url or ""
        
        # 初始化模型调用器
        self.reasoning_caller = LocalModelCaller(reasoning_config or REASONING_MODEL_CONFIG)
        self.grounder_caller = LocalModelCaller(grounder_config or GROUNDER_MODEL_CONFIG, is_grounder=True)
        
        # 执行历史
        self.steps: List[StepInfo] = []
        self.current_step = 0
        self.is_finished = False
        self.final_answer = ""
        self.last_screenshot = ""  # 保存最后一张截图用于评估
        
        # Todo list 用于跟踪任务进度，避免循环
        self.todo_list: List[Dict[str, Any]] = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    def get_reflector_domain_specific_tips(self) -> str:
        """根据当前页面URL，获取 Reflector 专用的领域专家提示"""
        current_url = self.page.url.lower()
        
        # Shopping Admin (7780)
        if "7780" in current_url:
            return """
# Adobe Commerce Documentation Expert Knowledge (Your Extraction Goal)
When analyzing Adobe Commerce documentation or admin interface, your main goal is to extract actionable information. **You must look for and extract the following key information from the page:**
- **Navigation Paths:** Look for text like **"On the Admin sidebar, go to..."** or menu items (e.g., `Sales > Orders`).
- **Data/Page Descriptions:** What information a page or report contains.
- **Alternative Paths:** Sometimes information can be found in multiple places.
- **Operational Tips:** Look for specific steps or advice on how to use features more efficiently.
    - **Example:** For calculating comprehensive sales over two months, an efficient tip is to select `Year` for Period, then filter by `From/To`, rather than selecting `Month` or `Day` and calculating manually. To query sales for a single month, please use the `Month` selection.
    - **Order Status Selection:** If the user request includes a description of the order status (e.g., success/completed, cancelled, closed, etc.), you MUST make a selection for the order status. This option is typically found in the "Order Status" or "Status" dropdown menu.
"""
        return ""

    def get_domain_specific_tips(self) -> str:
        """根据当前页面URL，获取领域特定的专家提示"""
        current_url = self.page.url.lower()
        tips_list = []

        # 合并后的 shopping_admin 提示
        if "7780" in current_url:
            print(f"加载 shopping_admin 提示")
            tips_list.append(
'''
# Adobe Commerce Admin Operation Expert Strategy (Your Knowledge Base and Action Guide)
Base URL: https://experienceleague.adobe.com/en/docs/commerce-admin/ (For documentation query)

**1. Core Navigation Paths (Keyword → Path):**
   - **Order Status:** `Sales -> Orders`
   - **Invoices:** `Sales -> Invoices`
   - **Reviews:** `Marketing -> All reviews`
   - **Search Terms:** `Reports -> Search Terms`
   - **Bestsellers:** `Reports -> Bestsellers`
   - **Products (Used to get and edit product attribute info):** `Catalog -> Products`
   - **Customers (Used to get and edit user info):** `Customers -> ALL Customers`
   - **Pages (Used to get page attributes like Title):** `Content -> Pages`
   - **Themes (Used to view store theme info):** `Content -> Themes`
   - **Reports Hierarchy:**
     - **Marketing:** `Reports -> Marketing`
     - **Sales:** `Reports -> Sales -> (Orders, Tax, Invoiced, Shipping, Refunds, Coupons)`
     - **Customers:** `Reports -> Customers -> (Order Total, Order Account, New, Wish Lists, Segments)`
     - **Products:** `Reports -> Products -> (Views, Bestsellers, Low Stock, Ordered, Downloads)`


**2. Common Workflows:**
   - **Check Sales:**
     1. For calculating comprehensive sales or bestsellers over two months, an efficient tip is to select `Year` for Period, rather than selecting `Month` or `Day` and calculating manually. To query sales or bestsellers for a single month, please use the `Month` selection. **This type of tip is very valuable and must be recorded.**
     2. Directly **Input (Type)** the start and end dates of the quarter in the `From` and `To` fields, instead of using the calendar selection (e.g., directly input, 01/01/22 to 03/31/22).
     3. Click `Show Report`.
     4. Scroll down to view the report.
   - **Search Product Reviews (e.g., "tanks"):**
     1. Navigate to `Marketing -> All reviews`.
     2. **Keywords**: Note that it may not be the full product name in the request. Keep keywords minimal, try searching the first word first, search for brand if available (e.g., "tank").
     3. Fill in the Product name in the Product column, do not use SKU.

**3. Critical Rules:**
   - **If valid information is not found**, try different path methods multiple times.
   - **Data Year Limitation:** The store backend only has data for **2022** and the **first five months of 2023**. Do not try to query data outside this range.
   - **Date Format:** All date input boxes must use the format `MM/DD/YY` (e.g., `03/31/22`).
   - **Completeness (Visual vs Data):** Just because a table fits in the screen doesn't mean it's complete.
     - **Mandatory Scroll Check**: After a report is displayed, you **MUST** execute at least one `scroll` action to verify integrity.
     - If the user asks for "top N" items and you see fewer than N, assume data is hidden below the fold.
     - **Action**: Unless you see "Page 1 of 1" or "End of results", strictly prioritize `scroll` to the very bottom to ensure all potential rows are rendered.
   - **Start Filling Rule:** When filling in From, **To must also be filled**. From and To can be the same, From and To include the equal case.
   - **Report Display Complete:** When you need to retrieve information in the report, you must scroll the page or content. The report may be at the bottom and make sure you scroll to the bottom of the page to see the complete report. **Unless you have clear information, all information to be retrieved must be loaded before stopping scrolling.**
   - **Report Sorting:** Report headers can be clicked to sort different options, which is very useful for requests like Newest and Oldest.
   - **Keywords:** When filling in keywords, keep the input keywords as few as possible, try to use one word. If the search fails, try using other keywords.
   - **Scroll:** Always scroll down to bottom when viewing a list/report. Do not assume data is complete without scrolling.
   - **Quantity Check**: If the user requires a specific number of items (e.g., "top 3") but you found fewer (e.g., 2), you must **strictly** verify if the page can be scrolled further.
   - **Select Option:** All UI elements with a **downward triangle symbol** are dropdown menus and should be operated using **select_option** (e.g., Order, Period etc.). 
   - **Order Status:** If the user request includes a **Specified** order status (e.g., success/completed, cancelled, closed, etc.), you MUST make a selection for the order status. When you see the "Order Status" dropdown menu, you can firstly select "Specified" to see the options and then select the specific order status.

# Core Task Guidelines
*   **[Highest Priority] Apply Expert Strategy:** You **must** use the built-in **"Adobe Commerce Admin Operation Expert Strategy"** to guide your plan. Your `thought` process must explicitly state how you apply the strategy.
    *   **Step 1:** Analyze `user_query` to extract core intent.
    *   **Step 2:** Refer to **"Core Navigation Paths"** and **"Common Workflows"** to find the most matching strategy.
    *   **Step 3:** During execution, strictly abide by **"Critical Rules"** (such as date format, year limitation, completeness, etc.).
    *   **Example:** If `user_query` is "Customers are unsatisfied with tanks products, check reviews", your thinking process should be: "User intent is to query negative reviews about 'tanks'. According to expert strategy, I should execute the 'Search Product Reviews' workflow. Step 1 is to navigate to `Marketing -> All reviews`. I will generate a `click` action to click 'Marketing' in the left menu."
*   The instruction output to the executor (Grounder) must be a single atomic operation for the next step.
*   If the page does not change after entering text, try clicking nearby "Search", "Show Report" buttons.

'''
            )


        if ":7770" in current_url: # 假设 shopping 运行在 7770 端口
            print(f"加载 shopping 提示")
            tips_list.append(
                shopping_navigation_prompt+"\n\n"
f'''
** Target URL: **
{self.original_target_url}
If |AND| is in the target url, the first url has already been loaded in the initial page and **you need to navigate to the second url firstly.**.

**Order Operation Rule:**
For order operations (e.g., buying or canceling), do NOT just add items to the cart. You MUST proceed through the entire flow: go to the shopping cart, checkout, and complete the order.

**show products under a price:**
When asked to "show me products under a price" the benchmark evaluates based on the url. So just ensure the price is under the price asked for in the url. For example, when asked "Show me products under $25 in \\"women shoes\\" category", you go to "http://{self.REPLACE_WITH_YOUR_HOST}:7770/clothing-shoes-jewelry/women/shoes.html?price=0-25" derectly.

**For SHOPPING/BROWSING tasks**: 
When asked to browse products in a particular category, navigate using the dropdown menus (not search) when possible. This may require hovering over nested dropdowns (e.g., hover over "Electronics" → hover over "Computers" → click "Laptops"). Use the hover tool to reveal these nested menus before clicking.

**For market survy tasks**: 
When asked to doing a market survey, you must end with clicking to enter the product details page. For example, when asked "I am doing a market survey for one stop market, show me the most expensive product from PS4 accessories category", you must go to "http://{self.REPLACE_WITH_YOUR_HOST}:7770/astro-gaming-a50-wireless-headset-base-station-gen-4-compatible-with-ps5-ps4-pc-mac-black-silver.html" finally.

**For viewing reviews tasks**
You must view all the conmplete reviews. If there are multiple pages of comments, you must click on each page to ensure that you see all the reviews(Note: the page number is at the bottom of the page). For every pages, you must scroll to the bottom of the page to see all the complete reviews.

**For Ordering Time tasks**
When asked about the ordering time for a specific product. You must open orders from near to far according to time, and once you find the corresponding order for the target product, give the answer. For example, when asked "Tell me when I last ordered my body butter?", you should first click "my account", and then click "my order", and then click "view order" of orders with "Complete" status. If you can't find the target product in this page, you must click the next page number and repeat the above process. Once you find the corresponding order for the target product, give the answer.

**For Draft a refund message tasks**
When asked 'Draft a refund message via their "contact us" form for the {{product}} I bought {{time}}. It broke after three days of use. The shop requires the order id, the reason and the amount to refund in the message. Don't submit yet', you must draft a message include Order Id, the reason, and the amount to refund. Specially, the reason is "It broke after three days of use.", the order id is that order number you find by the given time, and the amount is the price of the given product. For make sure of the amount to refund, you must click the "View Order" and find the true product.

** Reddit Website: **
The URL of the Reddit Website share the same ip of Target URL, but the port is 9999.
If you want to navigate to a subreddit of the Reddit Website, you directly goto the subreddit URL: http://{self.REPLACE_WITH_YOUR_HOST}:9999/f/<subreddit_name>.
'''
            )
        # 合并后的 openstreet map 提示
        if ":3000" in current_url: # 假设 openstreet map 运行在 3000 端口
            tips_list.append(                   
                """
- **For the OpenStreetMap website:**
  - **Critical Note:** On the initial OpenStreetMap page, after searching for a location, the search results will appear as links. **These links are invalid and MUST NOT be clicked!** You can only reference the link text to get detailed address information.
  - You must first extract or infer a specific location name from the user's `intent`. For example, a description like `the capital of New York State` must be replaced with `Albany` for searching or navigation. OpenStreetMap only supports searches for place names and coordinates in DD format. Vague queries like `airports that are within 50 km to CMU` are not supported and must be converted to a precise airport name, such as `Pittsburgh International Airport`, before searching. If a search fails, try modifying the query by adding or removing city, county, or neighborhood names, then search again.
  - OpenStreetMap has two search boxes: one on the initial page and another on the directions page. You must choose the appropriate search box based on the specific query.
  - For queries about distance, walking time, or driving time between two places, use the directions feature to get the answer. The entry point for the directions interface is the blue arrow button located to the right of the blue `Go` button on the initial page.
  - For queries like `Which US states border Vermont?`, since the map data may be incomplete, simply search for `Vermont` in the search box and then answer directly. Your answer should only consider states with a direct land border.
  - To select the mode of transport on the directions page, use the select_option action for the dropdown menu that appears after clicking. You must strictly use the options provided by the website and not guess. The only valid transport options are Car (OSRM), Bicycle (OSRM), and Foot (OSRM).
  - For queries asking if it's possible to travel from `place1` to `place2` within a specified time, you must use the navigation feature to determine the answer.
  - For queries about the walking or driving time from place1 to place2, you must use the directions feature to obtain the answer. The answer should be formatted as X hours Y minutes, eg, 1 hour 25 minutes.
  - To zoom in or out on the map, **do not use `scroll up` or `scroll down`**. You must use the `zoom_in` and `zoom_out` actions, passing the desired zoom level in the `level` parameter.
  - For queries asking for a `zip code`, search for the location on the initial page. The zip code can be found in the search result string. For example, if the search result for CMU is `Carnegie Mellon University, ..., 15232, United States`, you should directly output `15232`.
  - If you need the coordinates of a location, first search for it on the initial page. Then, find the target entry in the search results and hover your mouse over it. A location marker will appear on the map. Right-click on this marker and select `Show Address` to get the coordinates in Decimal Degrees (DD) format.
  - For queries about a detailed address (e.g., the address of an international airport), you must search for the location on the initial OpenStreetMap page. **Prioritize using the address from the resulting link text.** Even if you can visually locate the target on the map, you MUST still perform a search to use the official address from the search result. If the location cannot be found via search, use the following format as a fallback: `Institution/Place Name, Street Name, Neighborhood/District Name, City, County, State, ZIP Code, Country`. For example: `Carnegie Mellon University, Canterbury Lane, Shadyside, Pittsburgh, Allegheny County, Pennsylvania, 15232, United States`.
  - For queries like `the closest/nearest [place1] to [place2]`, combine your prior knowledge with map data. If you already know the specific name of the closest place, use that name directly. If you do not know, you must first search for `[place2]` and then visually scan the map to find the closest `[place1]`.
  - For queries like Pull up the description page of {{location}} on Map, the task is complete once you have searched for the location on the initial OpenStreetMap page.
  - On the directions page, if the Go button has no effect after you've entered the start and end points, it means one of the locations is invalid. You should correct the name and try again. To verify a location's name, you can search for it on the initial OpenStreetMap page first, and then use the confirmed name on the directions page.
  - When searching for locations, do not count coffee shops as restaurants.             
                """   
            )

        if ":9999" in current_url: # reddit
            tips_list.append(
 f"""
 - **For the Reddit website:**
    - To access a user's profile page, the link is http://{self.REPLACE_WITH_YOUR_HOST}:9999/user/{{username}}
    - To view your own profile page (forum), the link is http://{self.REPLACE_WITH_YOUR_HOST}:9999/user/MarvelsGrantMan136
    - To view all forums, the link is http://{self.REPLACE_WITH_YOUR_HOST}:9999/forums/all
    - To view a specific forum, the link is http://{self.REPLACE_WITH_YOUR_HOST}:9999/f/{{forum_name}} (however, if it results in a 404 error URL, please go to http://{self.REPLACE_WITH_YOUR_HOST}:9999/forums/all instead)
    - To view the comments on a post (submission), click directly on "xxx comments" or "no comments" below the post title. Do not click the title itself, as it may redirect you to the image link or another web link contained in the post. Conversely, if you want to view the post's image, click the title.
    - To edit your own post, click "Edit" below the title, not the title itself.
    - Reposting is not currently supported. You must submit a new post to do this. If the original post contains an image, you need to first open the image, record its link, and then paste it into the URL field when submitting the new post.
    - If a task mentions "top n posts," it refers to the top n posts based on the default sorting order (i.e., hot).
    - If a task mentions something like "in r/sports," it refers to the "sports" forum. The same logic applies to other forums.
    - The term "subreddit" is used to mean "forum"
    - To access GitLab, the link is http://{self.REPLACE_WITH_YOUR_HOST}:8023
    - To access a specific GitLab repository {{username}}/{{reponame}}, the link is http://{self.REPLACE_WITH_YOUR_HOST}:8023/{{username}}/{{reponame}}
"""
            )

        if ":8023" in current_url: # gitlab 
            tips_list.append(
f"""
- **For the gitlab website:**
    - URL-First Strategy is Key: For GitLab, your primary strategy should always be to directly construct a URL and use the goto action, as this is far more efficient than UI navigation. Only if a constructed URL fails should you resort to step-by-step GUI operations. When constructing URLs, ensure you use the correct ECS IP address (http://{self.REPLACE_WITH_YOUR_HOST}:8023).
    - Constructing Issue URLs:
        - All issue-related tasks can be solved with a direct URL. The structure is http://{self.REPLACE_WITH_YOUR_HOST}:8023/<username>/<reponame>/-/issues/?<parameters>.
        - Parameter Order: You must follow a strict parameter order: sort first, then state, then label_name.
        - Translate Intent to Terms: You must translate natural language from the intent into GitLab's specific terminology. For example, help needed must be converted to help wanted, and bug must be capitalized to BUG.
        - Specific Filters:
            - To Check out the most recent open issues, go to the URL .../issues/?sort=created_date&state=opened.
            - To List all opened issues that don't have any labels, use the filter .../issues/?label_name%5B%5D=None.
            - For intents like Display the list of issues... with labels related to questions, directly goto .../issues/?label_name%5B%5D=question.
    - Interacting with Merge Requests:
        - For tasks like Post "{{content}}" for the merge request..., after filling in the content, you must click the 'Comment' button. Do not click the 'Comment & close merge request' button unless explicitly instructed to do so.
        - For tasks like Go to the merge request on {{topic}} I have to review..., you must start by clicking the 'Merge requests' icon in the top-right corner of the dashboard. You must check both the 'Assigned to you' and 'Review requests for you' sections to find the correct MR. To decide on your reply, check the activity feed: if it only contains a system message (e.g., user2 assigned...), reply with a simple @ mention; if there is other text content from the author, reply with Thank you.
    - Major UI Workflows:
        - Forking All Repositories: For tasks requiring you to fork all of a user's repositories, you must follow a multi-step process: first, navigate to the user's profile page, then locate their list of personal projects, and finally, iterate through the list, forking each project one by one.
        - Searching for Repositories: The search function does not support the user/reponame format. To find a repository like kkroening/ffmpeg-python, you must search for ffmpeg-python only and then locate the project owned by kkroening in the search results.
        - Finding and Following Users: For tasks like Follow {{account_list}}..., if a directly constructed profile URL results in a 404 error, you must use the search bar on the dashboard. After searching, click on the 'Users' filter in the sidebar to locate the correct user profile.
        - Handling User/Repo Names (URL Construction Priority): The displayed username (e.g., Byte Blaze) might differ from the actual name used in the URL path. If unsure, first navigate by clicking to the repository page. Then, get the correct username from the page's URL. Your next priority is to use this correct username to construct a new goto URL. Only if that new URL fails should you fall back to further GUI operations.
    - Cloning Repositories:
        - To find the clone URL, first check the project's README. If not there, click the "Clone" button. Use hscroll to scroll horizontally if the full URL is not visible.
        - For questions like Show me the command to clone {{repo}} with SSH, you must find the SSH clone URL and then replace the IP address with metis.lti.cs.cmu.edu in the final command. For example, git clone ssh://git@{self.REPLACE_WITH_YOUR_HOST}:2222/... must be changed to git clone ssh://git@metis.lti.cs.cmu.edu:2222/....
    - Finding Your Assigned Items:
        - To find issues, merge requests, or to-do items assigned to you, use the icons in the top-right corner of the dashboard.
        - For complex queries like Open my latest created issue that has <keyword> in its title, you can either click the "Issues" icon and search, or directly goto a URL like http://{self.REPLACE_WITH_YOUR_HOST}:8023/dashboard/issues?scope=all&state=opened&assignee_username=<your_username>&search=<keyword>.
    - Specific Answer Formatting:
        - For questions about people (who has made the most contributions, who else has access), your final answer must only be the person's name(s).
        - For tasks like ...to check if it is closed, you must first navigate to the specific issue's page (e.g., .../-/issues/719), not the search results page. Your final answer must then be only "Yes" or "No".
        - For tasks like Create a repo named ... with movies directed by ... in a README file, you must list all qualifying movies in the README. The list should contain only the movie titles, with no other information like release years.
    - Misc UI Tips:
        - To find public repositories, use the Explore section on the main page.
        - To get a precise timestamp from relative times (e.g., "updated 2 years ago"), hover your mouse over the text.
        - When setting your status to 'Busy', do not check the 'Set yourself as busy' checkbox. Instead, you must type Busy directly into the status input field.
- ** For Cross-Website Reddit Queries:**
        - For queries related to subreddits, such as "...URLs of the 5 most recent posts from the movies?" or "the most active {{num}} DIY ideas on DIY subreddit?", you must find the answers on the internally deployed Reddit website at "http://{self.REPLACE_WITH_YOUR_HOST}:9999. For each qualifying result, the required output is the URL of its comments page, for example: http://{self.REPLACE_WITH_YOUR_HOST}:9999/f/news/129905/ohio-man-charged-for-using-molotov-cocktails-to-attack".
        - For the Reddit website:
            - To access a user's profile page, the link is http://{self.REPLACE_WITH_YOUR_HOST}:9999/user/username
            - To view your own profile page, the link is  http://{self.REPLACE_WITH_YOUR_HOST}:9999/user/MarvelsGrantMan136
            - If a user_query mentions "top n posts," it refers to the top n posts based on the default sorting order (i.e., hot).
            - If a user_query mentions something like "in r/sports," it refers to the "sports" forum. The same logic applies to other forums.
            - The term "subreddit" is used to mean "forum"
            - To view all forums, the link is http://{self.REPLACE_WITH_YOUR_HOST}:9999/forums/all, if the user_query doesn't specify a forum, please go to this url to find the most relevant forum.
            - To view a specific forum, the link is http://{self.REPLACE_WITH_YOUR_HOST}:9999/f/forum_name (however, if it results in a 404 error URL, please go to http://{self.REPLACE_WITH_YOUR_HOST}:9999/forums/all instead)
            - To view the comments on a post (submission), click directly on "xxx comments" or "no comments" below the post title. Do not click the title itself, as it may redirect you to the image link or another web link contained in the post. Conversely, if you want to view the post's image, click the title.
            - To edit your own post, click "Edit" below the title, not the title itself.
            - Reposting is not currently supported. You must submit a new post to do this. If the original post contains an image, you need to first open the image, record its link, and then paste it into the URL field when submitting the new post.
"""

            )
        # 如果没有匹配的提示，返回默认值
        if not tips_list:
            return "None"
        
        # 将所有匹配的提示合并成一个字符串
        return "\n".join(tips_list)

    def get_summary_tips(self) -> str:
        """根据当前页面URL，获取用于生成最终答案的领域特定提示"""
        current_url = self.page.url.lower()
        tips_list = []

        # OpenStreetMap 网站的 Summary 提示
        if ":3000" in current_url:
            tips_list.append(
                """
                - **For OpenStreetMap tasks:** 
                    - For general knowledge queries like `Which US states border Vermont`, answer directly based on your geographical knowledge, using the map in the screenshot for confirmation. Your answer should only consider states with a direct land border.
                    - For questions about a detailed address, prioritize using the address found in the text of the OpenStreetMap search result link (which should be visible in the final screenshots). If no search result is found in the history, use your prior knowledge to construct the address in the following format as a fallback: Institution/Place Name, Street Name, Neighborhood/District Name, City, County, State, ZIP Code, Country. For example: `Carnegie Mellon University, Canterbury Lane, Shadyside, Pittsburgh, Allegheny County, Pennsylvania, 15232, United States`.
                    - For zip code queries, prioritize extracting the code directly from the OpenStreetMap search result visible in the screenshot. For example, extract `15232` from `Carnegie Mellon University, ..., 15232, United States`.
                    - For queries about coordinates in Decimal Degrees (DD) format, prioritize extracting them from the URL of the search result or from information displayed directly on the page (visible in the screenshots). If the coordinates cannot be found in the provided context, use your prior knowledge to provide the answer. The final coordinates must be rounded to three decimal places. For example, if OpenStreetMap provides `40.46081, -79.94668`, your output must be `40.460, -79.946`.
                    - When searching for locations, do not count coffee shops as restaurants.
                    - The travel time displayed in OpenStreetMap's directions is in an HH:MM format (e.g., 01:25, meaning 1 hour and 25 minutes). You must convert this into the X h Y min format for your final answer (e.g., 1 h 25 min).
                    - For queries like Measure distance between {{location/address_1}} and {{location/address_2}} by walking, you should only return the distance from the navigation results, for example, 1.2km.  
                    - Strict Format Matching**: If the example shows "557m", please use the exact same format. If it is name information like product user, return only the name in the answer, do not return other information.
                """
            )
        
        # # GitLab 网站的 Summary 提示
        if ":8023" in current_url:
            tips_list.append(
                """
- **For the gitlab website:**
    - Specific Answer Formatting:
        - Output Formatting for Clone Commands: For questions like Show me the command to clone <repo> with SSH, you must find the SSH clone URL and then replace the IP address with `metis.lti.cs.cmu.edu` in the final command. For example, `git clone ssh://git@{self.REPLACE_WITH_YOUR_HOST}:2222/yjlou/2019-nCov.git` must be changed to `git clone ssh://git@metis.lti.cs.cmu.edu:2222/yjlou/2019-nCov.git`.
        - For questions about people, like `who has made the most contributions` or `who else has access`, your final answer must only be the person's name(s). Do not include any other text.
        - For tasks like `...to check if it is closed`, your final answer must be only "Yes" or "No". Do not include any other text.
        - For queries related to subreddits, such as "...URLs of the 5 most recent posts from the movies?" or "the most active {{num}} DIY ideas on DIY subreddit?", you must find the answers on the internally deployed Reddit website at "http://{self.REPLACE_WITH_YOUR_HOST}:9999. For each qualifying result, the required output is the URL of its comments page, for example: http://{self.REPLACE_WITH_YOUR_HOST}:9999/f/news/129905/ohio-man-charged-for-using-molotov-cocktails-to-attack".
                """
            )
        # For Shopping
        if "7770" in current_url:
            tips_list.append(
                """
═══════════════════════════════════════════════════════════════════════
Answer Format (Very Important):
═══════════════════════════════════════════════════════════════════════

When you generate the final answer, please follow these principles:

1. **Strict Format Matching**: If the example shows "557m", please use the exact same format. If it is name information like product user, return only the name in the answer, do not return other information.
2. **Provide Complete Answer**: Include enough context so that the answer can stand alone.
3. **Add Reasoning Appropriately**: For questions requiring judgment (yes/no, status check, comparison), include short context or reasoning next to the answer.
4. **Accurate Terminology**: When copying text, use the exact wording from the source file.
5. **Ties**: When involving sorting or Top N questions, if there are numerical ties, you must list **all tied candidates** in the answer, even if this causes the final result quantity to exceed N. Please be sure to **carefully check the values**, do not just intercept the first N rows of the report, because the sorting of items with the same value in the report may be random, you must check if there are more items with the same value ranked behind.
6. When asked to return the answer in MM:COUNT format, please return it like this: "January: 1". It expects MM to be the explicit name of the month rather than a number.
7. When asked how much it costs, return only the decimal. Therefore, if the item costs $7.50, return "7.50"; if it costs $0, return "0".
8. When asked about configuration, return 2x2 instead of 2*2.
9. If there are multiple matching entries for amount-based questions, please list each amount in the reasoning and ensure that the final answer string contains the aggregated total satisfying the query (e.g., sum of all matching refunds).

                """
            )
        # For Shopping Admin
        if "7780" in current_url:
            tips_list.append(
                """
═══════════════════════════════════════════════════════════════════════
Answer Format (Very Important):
═══════════════════════════════════════════════════════════════════════

When you generate the final answer, please follow these principles:

1. **Strict Format Matching**: If the example shows "557m", please use the exact same format. If it is name information like product user, return only the name in the answer, do not return other information.
2. **Provide Complete Answer**: Include enough context so that the answer can stand alone.
3. **Add Reasoning Appropriately**: For questions requiring judgment (yes/no, status check, comparison), include short context or reasoning next to the answer.
4. **Accurate Terminology**: When copying text, use the exact wording from the source file.
5. **Ties**: When involving sorting or Top N questions, if there are numerical ties, you must list **all tied candidates** in the answer, even if this causes the final result quantity to exceed N. Please be sure to **carefully check the values**, do not just intercept the first N rows of the report, because the sorting of items with the same value in the report may be random, you must check if there are more items with the same value ranked behind.
6. When asked to return the answer in MM:COUNT format, please return it like this: "January: 1". It expects MM to be the explicit name of the month rather than a number.
7. When asked how much it costs, return only the decimal. Therefore, if the item costs $7.50, return "7.50"; if it costs $0, return "0".
8. When asked about configuration, return 2x2 instead of 2*2.
9. If there are multiple matching entries for amount-based questions, please list each amount in the reasoning and ensure that the final answer string contains the aggregated total satisfying the query (e.g., sum of all matching refunds).

                """
            )
        # 如果没有匹配的提示，返回默认值
        if not tips_list:
            return "None"
        
        return "\n".join(tips_list)

    def get_recent_steps_detail(self, n: int = 3) -> str:
        """获取最近 n 步的执行细节"""
        if not self.steps:
            return "无历史执行记录"
        
        recent = self.steps[-n:]
        details = []
        for step in recent:
            detail = f"Step {step.step_id}:\n"
            detail += f"  Action: {json.dumps(step.action, ensure_ascii=False)}\n"
            if step.error_msg:
                detail += f"  Error: {step.error_msg}\n"
            details.append(detail)
        
        return "\n".join(details)
    
    def get_execution_path(self) -> str:
        """获取执行路径描述"""
        if not self.steps:
            return "无执行路径"
        
        path_items = []
        for step in self.steps:
            action = step.action
            action_type = action.get('action_type', 'unknown')
            path_items.append(f"Step {step.step_id}: {action_type}")
        
        return " -> ".join(path_items)
    
    def get_marked_notes(self) -> str:
        """获取已收集的备注信息"""
        notes = []
        for step in self.steps:
            if step.note and step.note.strip():
                notes.append(f"Step {step.step_id}: {step.note}")
        
        if not notes:
            return "None"
        
        return "\n".join(notes)
    
    def get_todolist_status(self) -> str:
        """获取 todo-list 状态"""
        if not self.todo_list:
            return "None (Create a todo list based on the user_query)"
        
        lines = []
        for item in self.todo_list:
            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(item.get("status", "pending"), "⏳")
            lines.append(f"{status_icon} [{item.get('id', '?')}] {item.get('description', '')}")
        
        return "\n".join(lines)
    
    async def call_reflector(self, prev_screenshot: str, curr_screenshot: str, 
                              error_msg: str = "") -> Dict[str, Any]:
        """
        调用 Reflector 模块
        
        和 control_agent 一致：
        1. 使用两张截图（上一步 + 当前）
        2. 传入完整的上下文信息
        """

        # 获取 Reflector 专用领域知识
        domain_tips = self.get_reflector_domain_specific_tips()
        prompt = REFLECTION_PROMPT.format(
            user_query=self.user_query,
            html_simplify="None",  # 当前版本暂不提取 HTML
            tips=domain_tips,
            recent_10_step_details=self.get_recent_steps_detail(10),
            marked_note=self.get_marked_notes(),
            current_path=self.get_execution_path(),
            todolist_status=self.get_todolist_status()
        )
        
        # 和 control_agent 一致：使用两张截图 [上一步截图, 当前截图]
        images = []
        if prev_screenshot:
            images.append(prev_screenshot)
        if curr_screenshot:
            images.append(curr_screenshot)
        
        # 如果没有上一步截图，只用当前截图
        if not images and curr_screenshot:
            images = [curr_screenshot]
            
        result = await self.reasoning_caller.call(prompt, images)
        
        logger.debug(f"[Reflector] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    async def call_planner(self, screenshot: str, reflection_signal: str = "") -> Dict[str, Any]:
        """调用 Planner 模块"""
        # 在调用 Planner 前，动态获取当前页面的 tips
        domain_tips = self.get_domain_specific_tips()

        prompt = PLANNER_PROMPT.format(
            user_query=self.user_query,
            tips=domain_tips,  # 在这里注入动态生成的 tips
            todo_list_status=self.get_todolist_status(),
            reflection_signal=reflection_signal or "无",
            marked_note=self.get_marked_notes(),
            recent_10_step_details=self.get_recent_steps_detail(10)
        )
        
        result = await self.reasoning_caller.call(prompt, [screenshot])
        
        logger.debug(f"[Planner] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    
    async def call_grounder(self, screenshot: str, instruction: str) -> Dict[str, Any]:
        """
        调用 Grounder 模块
        
        和 control_agent 完全一致：
        1. 构建包含历史截图的 image_url_list
        2. 构建 previous_actions 历史动作
        3. 拼接完整的 prompt
        4. 上传图片到 AFTS 获取 URL
        5. 调用 codebot API
        """
        # 获取 AFTS 工具
        afts_tool = get_afts_tool()
        
        # === 1. 构建 image_url_list 和 previous_actions (和 control_agent 一致) ===
        image_url_list = []
        previous_actions = {}
        
        # 当前截图必须添加
        current_image_url = afts_tool.upload_base64_image(screenshot, f"screenshot_current_{int(time.time())}.png")
        if not current_image_url:
            logger.error("[Grounder] Failed to upload current screenshot")
            return {}
        image_url_list.append(current_image_url)
        
        # 获取最近的历史步骤 (最多取最近2步，加上当前共3张图)
        if len(self.steps) >= 2:
            # 添加倒数第2步
            step_n2 = self.steps[-2]
            if step_n2.screenshot_base64:
                img_url = afts_tool.upload_base64_image(step_n2.screenshot_base64, f"screenshot_step_{step_n2.step_id}.png")
                if img_url:
                    image_url_list.append(img_url)
                    previous_actions[1] = {
                        "name": step_n2.action.get('action_type', ''),
                        "arguments": step_n2.action.get('params', {})
                    }
            
            # 添加倒数第1步
            step_n1 = self.steps[-1]
            if step_n1.screenshot_base64:
                img_url = afts_tool.upload_base64_image(step_n1.screenshot_base64, f"screenshot_step_{step_n1.step_id}.png")
                if img_url:
                    image_url_list.append(img_url)
                    previous_actions[2] = {
                        "name": step_n1.action.get('action_type', ''),
                        "arguments": step_n1.action.get('params', {})
                    }
        elif len(self.steps) >= 1:
            # 只有1步历史
            step_n1 = self.steps[-1]
            if step_n1.screenshot_base64:
                img_url = afts_tool.upload_base64_image(step_n1.screenshot_base64, f"screenshot_step_{step_n1.step_id}.png")
                if img_url:
                    image_url_list.append(img_url)
                    previous_actions[1] = {
                        "name": step_n1.action.get('action_type', ''),
                        "arguments": step_n1.action.get('params', {})
                    }
        
        # === 2. 构建 description (和 control_agent 一致，包含用户目标) ===
        description = f"你的总目标是：{self.user_query}；当前目标是：{instruction}"
        
        # === 3. 构建 prompt (和 control_agent 的 _get_prompt 一致) ===
        prompt = GROUNDER_PROMPT.format(instruction=description)
        
        # 拼接历史动作 (和 control_agent 一致)
        if previous_actions:
            prompt += "Previous actions:\n"
            for step_id, action in previous_actions.items():
                action_str = f"Step {step_id}:\n<image>\n{json.dumps(action, ensure_ascii=False)}"
                prompt += action_str + "\n\n"
        
        logger.info(f"[Grounder] image_url_list={len(image_url_list)}, previous_actions={previous_actions}")
        
        # === 4. 调用 Grounder API ===
        result = call_grounder_api(image_url_list, prompt)
        
        if result is None:
            result = {}
        
        # === 5. 坐标还原 (和 control_agent/qwen_72b_multi_image.py 一致) ===
        # Grounder 返回的坐标是基于压缩后的图像，需要还原到原始图像坐标
        coords = None
        if 'coords' in result:
            coords = result['coords']
        elif 'arguments' in result and 'coords' in result.get('arguments', {}):
            coords = result['arguments']['coords']
        
        if coords and len(coords) >= 2 and len(image_url_list) > 0:
            # 使用第一张图（当前截图）的尺寸进行坐标还原
            current_image_url = image_url_list[0]
            reverted_coords = revert_coords(coords, current_image_url)
            
            # 更新 result 中的坐标
            if 'coords' in result:
                result['coords'] = reverted_coords
            elif 'arguments' in result and 'coords' in result.get('arguments', {}):
                result['arguments']['coords'] = reverted_coords
            
            logger.info(f"[Grounder] 坐标已还原: {coords} -> {reverted_coords}")
        
        logger.debug(f"[Grounder] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    async def call_grounder_single(self, screenshot: str, instruction: str) -> Dict[str, Any]:
            afts_tool = get_afts_tool()
            
            # === 1. 构建 image_url_list 和 previous_actions (和 control_agent 一致) ===
            image_url_list = []
            previous_actions = {}
            
            # 当前截图必须添加
            current_image_url = afts_tool.upload_base64_image(screenshot, f"screenshot_current_{int(time.time())}.png")
            if not current_image_url:
                logger.error("[Grounder] Failed to upload current screenshot")
                return {}
            image_url_list.append(current_image_url)
            # === 3. 构建 prompt (和 control_agent 的 _get_prompt 一致) ===
            prompt = GROUNDER_PROMPT.format(instruction=instruction)
            
        
            #logger.info(f"[Grounder] image_url_list={len(image_url_list)}, previous_actions={previous_actions}")
            
            # === 4. 调用 Grounder API ===
            result = call_grounder_api(image_url_list, prompt)
            
            if result is None:
                result = {}
            
            # === 5. 坐标还原 (和 control_agent/qwen_72b_multi_image.py 一致) ===
            # Grounder 返回的坐标是基于压缩后的图像，需要还原到原始图像坐标
            coords = None
            if 'coords' in result:
                coords = result['coords']
            elif 'arguments' in result and 'coords' in result.get('arguments', {}):
                coords = result['arguments']['coords']
            
            if coords and len(coords) >= 2 and len(image_url_list) > 0:
                # 使用第一张图（当前截图）的尺寸进行坐标还原
                current_image_url = image_url_list[0]
                reverted_coords = revert_coords(coords, current_image_url)
                
                # 更新 result 中的坐标
                if 'coords' in result:
                    result['coords'] = reverted_coords
                elif 'arguments' in result and 'coords' in result.get('arguments', {}):
                    result['arguments']['coords'] = reverted_coords
                
                logger.info(f"[Grounder] 坐标已还原: {coords} -> {reverted_coords}")
            
            logger.debug(f"[Grounder] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
            return result
    
    async def call_summary(self, current_screenshot: str, max_screenshots: int = 5) -> Dict[str, Any]:
        """调用 Summary 模块生成最终答案
        
        Args:
            current_screenshot: 当前截图的 base64 编码
            max_screenshots: 最多使用多少张历史截图（默认5张）
        
        Returns:
            包含 answer 和 success 的字典
        """
        execution_history = self.get_recent_steps_detail(10)
        
        # 收集最后 N 张截图
        screenshots = []
        
        # 从 steps 中获取历史截图
        recent_steps = self.steps[-max_screenshots:] if len(self.steps) >= max_screenshots else self.steps
        for step in recent_steps:
            if step.screenshot_base64:
                screenshots.append(step.screenshot_base64)
        
        # 添加当前最新截图
        if current_screenshot:
            screenshots.append(current_screenshot)
        
        # 确保至少有一张截图
        if not screenshots:
            screenshots = [current_screenshot] if current_screenshot else []
        
        # 限制最多 max_screenshots 张
        if len(screenshots) > max_screenshots:
            screenshots = screenshots[-max_screenshots:]
        
        # 获取领域特定的 tips
        summary_tips = self.get_summary_tips()
        
        # 获取收集的笔记
        marked_note = self.get_marked_notes()
        
        prompt = SUMMARY_PROMPT.format(
            user_query=self.user_query,
            tips=summary_tips,
            execution_history=execution_history,
            marked_note=marked_note,
            screenshot_count=len(screenshots)
        )
        
        logger.info(f"[Summary] 使用 {len(screenshots)} 张截图生成最终答案")
        result = await self.reasoning_caller.call(prompt, screenshots)
        
        logger.debug(f"[Summary] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result
    
    async def run(self) -> Dict[str, Any]:
        """执行完整的任务流程"""
        start_time = time.time()
        
        # 初始截图
        prev_screenshot = ""
        curr_screenshot = await screenshot_to_base64(self.page)
        self.last_screenshot = curr_screenshot  # 保存最后一张截图用于评估
        await save_screenshot(self.page, os.path.join(self.output_dir, "images", "screenshot_0.png"))
        
        logger.info(f"[Agent] Starting task: {self.user_query[:100]}...")
        

        while self.current_step < self.max_steps:
            self.current_step += 1
            elapsed = time.time() - start_time
            
            if elapsed > self.timeout:
                logger.warning(f"[Agent] Task timeout after {elapsed:.1f}s")
                break
                
            logger.info(f"[Agent] Step {self.current_step}")
            
            try:
                # 1. 调用 Reflector
                reflection_result = await self.call_reflector(
                    prev_screenshot, curr_screenshot,
                    self.steps[-1].error_msg if self.steps else ""
                )
                
                # 检查任务是否完成
                if reflection_result.get('is_task_done', False):
                    logger.info("[Agent] Task marked as done by Reflector")
                    summary_result = await self.call_summary(curr_screenshot)
                    self.final_answer = summary_result.get('answer', '')
                    self.is_finished = True
                    break
                
                # 更新 todo_list（从 Reflector 结果中获取）
                if reflection_result.get('todo_list'):
                    self.todo_list = reflection_result['todo_list']
                    logger.info(f"[Agent] Todo list updated: {len(self.todo_list)} items")
                
                # 2. 调用 Planner
                reflection_signal = ""
                if reflection_result.get('instruction'):
                    inst = reflection_result['instruction']
                    if isinstance(inst, dict):
                        reflection_signal = f"级别: {inst.get('level', '')}, 内容: {inst.get('content', '')}"
                    else:
                        reflection_signal = str(inst)
                
                planner_result = await self.call_planner(curr_screenshot, reflection_signal)
                
                action_type = planner_result.get('action_type', '')
                action_attrs = planner_result.get('action_attributes', {}) or {}
                instruction = planner_result.get('instruction', '')
                
                if not action_type:
                    logger.warning("[Agent] Planner returned no action")
                    continue
                
                # 检查是否是终止动作
                if action_type.lower() in ['stop', 'finish', 'done', 'answer']:
                    logger.info("[Agent] Planner returned stop action")
                    summary_result = await self.call_summary(curr_screenshot)
                    self.final_answer = summary_result.get('answer', action_attrs.get('answer', ''))
                    self.is_finished = True
                    break
                
                # # 3. 对于需要坐标的动作，调用 Grounder
                # grounder_result = {}
                # if action_type in ['click', 'type', 'hover', 'scroll'] and instruction:
                #     grounder_result = await self.call_grounder(curr_screenshot, instruction)
                    
                #     # 合并坐标
                #     if 'coords' in grounder_result:
                #         action_attrs['coords'] = grounder_result['coords']
                #     elif 'arguments' in grounder_result and 'coords' in grounder_result['arguments']:
                #         action_attrs['coords'] = grounder_result['arguments']['coords']
                    
                #     if action_type == 'type':
                #         if 'content' not in action_attrs and 'text' in grounder_result:
                #             action_attrs['content'] = grounder_result['text']
                #         if 'content' not in action_attrs and 'arguments' in grounder_result:
                #             action_attrs['content'] = grounder_result['arguments'].get('text', '')
                
                # # 构建最终动作
                # action = {
                #     'action_type': action_type,
                #     'params': action_attrs
                # }
                
                # # 4. 执行动作
                # logger.info(f"[Agent] Executing: {action_type} with {action_attrs}")
                # error_msg = await execute_browser_action(self.page, action_type, action_attrs)


                # 3. 对于需要坐标的动作，调用 Grounder
                grounder_result = {}

                # 将 'select_option', 'right_click', 'zoom_in', 'zoom_out' 也视为需要坐标的动作
                if action_type in ['click', 'type', 'hover', 'scroll', 'select_option', 'right_click', 'zoom_in', 'zoom_out'] and instruction:
                    # 调用单图 Grounder（不带历史截图和历史动作，更轻量）
                    grounder_result = await self.call_grounder_single(curr_screenshot, instruction)
                    
                    # 合并坐标 (这部分逻辑完全不用动)
                    if 'coords' in grounder_result:
                        action_attrs['coords'] = grounder_result['coords']
                    elif 'arguments' in grounder_result and 'coords' in grounder_result['arguments']:
                        action_attrs['coords'] = grounder_result['arguments']['coords']

          
      
                # 构建最终动作 (这里的 action_type 依然是 Planner 最初给的 'select_option')
                action = {
                    'action_type': action_type,
                    'params': action_attrs
                }

                # 4. 执行动作
                logger.info(f"[Agent] Executing: {action_type} with {action_attrs}")
                error_msg = await execute_browser_action(self.page, action_type, action_attrs)



                # 5. 记录步骤信息 (保存当前截图的 base64，供 Grounder 使用历史截图)
                # 注意：先保存当前截图到 step_info，再更新 prev/curr_screenshot
                step_screenshot_base64 = curr_screenshot  # 保存执行动作前的截图
                prev_screenshot = curr_screenshot
                curr_screenshot = await screenshot_to_base64(self.page)
                self.last_screenshot = curr_screenshot  # 更新最后一张截图用于评估
                screenshot_path = await save_screenshot(
                    self.page, 
                    os.path.join(self.output_dir, "images", f"screenshot_{self.current_step}.png")
                )
                
                step_info = StepInfo(
                    step_id=self.current_step,
                    action=action,
                    screenshot_path=screenshot_path,
                    screenshot_base64=step_screenshot_base64,  # 保存截图 base64 供 Grounder 使用
                    page_url=self.page.url,
                    error_msg=error_msg,
                    reflection=reflection_result,
                    planner_output=planner_result,
                    grounder_output=grounder_result,
                    note=reflection_result.get('note', '') if reflection_result.get('mark_node', False) else ''
                )
                self.steps.append(step_info)
                
            except Exception as e:
                logger.error(f"[Agent] Step {self.current_step} error: {e}")
                traceback.print_exc()
                continue
        
        # 生成最终结果
        if not self.is_finished:
            try:
                logger.info("[Agent] Generating final answer...")
                summary_result = await self.call_summary(curr_screenshot)
                answer = summary_result.get('answer', '')
                if answer:
                    self.final_answer = answer
                    logger.info(f"[Agent] Final answer generated: {answer[:100]}")
            except Exception as e:
                logger.error(f"[Agent] Summary failed after 50 retries: {e}")
                # 50次都失败了，跳过这个测试样例
                logger.warning("[Agent] Skipping this task due to repeated failures.")
        
        # 保存轨迹 (兼容 webarena_online_eval.py 格式)
        trajectory = self._build_trajectory()
        
        trajectory_path = os.path.join(self.output_dir, 'trajectory.json')
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[Agent] Task completed. Steps: {self.current_step}, Answer: {self.final_answer[:100] if self.final_answer else 'N/A'}")
        
        return {
            'trajectory': trajectory,
            'final_answer': self.final_answer,
            'is_finished': self.is_finished,
            'total_steps': self.current_step,
            'steps': [asdict(s) for s in self.steps]
        }
    
    def _build_trajectory(self) -> List[Dict]:
        """构建与 webarena_online_eval.py 兼容的轨迹格式，同时保存模型输出"""
        trajectory = []
        
        # 添加初始观察
        trajectory.append({
            "type": "observation",
            "image_path": "images/screenshot_0.png"
        })
        
        # 处理每一步
        for step in self.steps:
            # 添加动作 (包含模型输出)
            trajectory.append({
                "type": "action",
                "action": step.action,
                "reflector_output": step.reflection,
                "planner_output": step.planner_output,
                "grounder_output": step.grounder_output,
                "page_url": step.page_url,
                "error_msg": step.error_msg
            })
            
            # 添加观察
            trajectory.append({
                "type": "observation",
                "image_path": f"images/screenshot_{step.step_id}.png"
            })
        
        # 如果有最终答案，添加到轨迹
        if self.final_answer:
            trajectory.append({
                "type": "generated_answer",
                "generated_final_answer": f"<|im_start|>assistant\n<answer>{self.final_answer}</answer>\n<|im_end|>"
            })
        
        return trajectory

    def get_evaluation_trajectory(self) -> List[Any]:
        """构建用于 evaluator 的轨迹格式 (包含 numpy 图片)"""
        traj = []
        for step in self.steps:
            # Observation (StateInfo)
            if step.screenshot_base64:
                try:
                    img_data = base64.b64decode(step.screenshot_base64)
                    image = Image.open(BytesIO(img_data))
                    observation = {"image": np.array(image), "text": ""}
                    state_info = {"observation": observation, "info": {"page": None, "url": step.page_url}}
                    traj.append(state_info)
                except Exception as e:
                    logger.warning(f"Failed to decode image for step {step.step_id}: {e}")
            
            # Action
            action_data = step.action
            flat_action = {
                "action_type": action_data.get('action_type'),
            }
            params = action_data.get('params', {})
            
            if 'coords' in params:
                flat_action['coords'] = np.array(params['coords'])
            
            if 'text' in params:
                flat_action['text'] = params['text']
            elif 'content' in params:
                flat_action['text'] = params['content']
            elif 'option' in params:
                flat_action['text'] = params['option']
                 
            flat_action['url'] = step.page_url
            
            traj.append(flat_action)

        # Final observation
        if self.last_screenshot:
            try:
                img_data = base64.b64decode(self.last_screenshot)
                image = Image.open(BytesIO(img_data))
                observation = {"image": np.array(image), "text": ""}
                state_info = {"observation": observation, "info": {"page": None, "url": self.page.url}}
                traj.append(state_info)
            except Exception as e:
                logger.warning(f"Failed to decode final image: {e}")
            
        return traj


# =============================================================================
# 工具函数
# =============================================================================

def get_REPLACE_WITH_YOUR_HOSTs(file_path: str) -> List[str]:
    """从 CSV 文件读取 ECS IP 列表"""
    ips = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                idx = header.index("公网 IP")
            except:
                idx = 1
            for row in reader:
                if len(row) > idx and row[idx].strip():
                    ips.append(row[idx].strip())
    except Exception as e:
        logger.error(f"Error reading ECS IPs: {e}")
    return ips


def load_cookies_for_task(target_url: str, webarena_auth_path: str, REPLACE_WITH_YOUR_HOST: str) -> Optional[Dict]:
    """根据任务 URL 加载对应的 cookies"""
    
    auth_dir = f"{webarena_auth_path}_{REPLACE_WITH_YOUR_HOST}/.auth"

    # 根据 URL 端口确定需要哪些站点的 cookies
    sites_needed = set()
    for port, site_name in SITE_PORT_MAP.items():
        if f":{port}" in target_url:
            sites_needed.add(site_name)
    
    # shopping 和 shopping_admin 共享 cookies
    if "shopping" in sites_needed or "shopping_admin" in sites_needed:
        sites_needed.update(["shopping", "shopping_admin"])
    
    if "gitlab" in sites_needed or 'reddit' in sites_needed:
        sites_needed.update(['gitlab','reddit'])

    if not sites_needed:
        logger.info(f"[{REPLACE_WITH_YOUR_HOST}] No login required for URL: {target_url}")
        return None
    
    # 构建 cookie 文件名列表
    cookie_files_to_try = []
    sorted_sites = sorted(list(sites_needed))
    
    # 尝试组合文件名
    cookie_files_to_try.append(f"{'.'.join(sorted_sites)}_state.json")
    
    # 尝试单站点文件名
    for site in sorted_sites:
        cookie_files_to_try.append(f"{site}_state.json")
    
    # 尝试加载
    for cookie_filename in cookie_files_to_try:
        cookie_file_path = os.path.join(auth_dir, cookie_filename)
        try:
            if os.path.exists(cookie_file_path):
                with open(cookie_file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "cookies" in data and len(data["cookies"]) > 0:
                        logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Loaded {len(data['cookies'])} cookies from {cookie_file_path}")
                        return data
        except Exception as e:
            logger.warning(f"[{REPLACE_WITH_YOUR_HOST}] Failed to read {cookie_filename}: {e}")
    
    logger.warning(f"[{REPLACE_WITH_YOUR_HOST}] Cookie file not found for {target_url}")
    return None


# =============================================================================
# 任务处理
# =============================================================================

async def process_single_task_local(
    task_config: Dict[str, Any],
    output_dir: str,
    target_url: str,
    actor: BrowserActor,  # 使用 BrowserActor 的浏览器
    cookies_content: Dict = None,
    reasoning_config: Dict = None,
    grounder_config: Dict = None,
    REPLACE_WITH_YOUR_HOST: str = None,
    config_file: str = None,  # 任务配置文件路径，用于评估
) -> bool:
    """
    处理单个任务
    
    使用 BrowserActor 的浏览器，而不是创建新的浏览器实例。
    这样可以复用 ssh_connect_and_refreshweb 创建的 cookies。
    """
    task_id = task_config.get('task_id', 'unknown')
    intent = task_config.get('intent', '')
    
    task_output_dir = os.path.join(output_dir, f"val_{task_id}")
    os.makedirs(task_output_dir, exist_ok=True)
    
    logger.info(f"[Task {task_id}] Starting on {target_url[:50]}...")
    
    try:
        # 使用 BrowserActor 的浏览器 (与 replay_and_evaluate.py 一致)
        browser = actor.browser_unit.browser
        
        # 创建新的 context
        context = await browser.new_context(
            viewport={"width": 1440, "height": 1200},
            ignore_https_errors=True,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        )
        
        # 如果有 cookies，添加到 context
        if cookies_content and "cookies" in cookies_content:
            try:
                await context.add_cookies(cookies_content["cookies"])
                cookie_domains = set(c.get('domain', 'N/A') for c in cookies_content["cookies"])
                logger.info(f"[Task {task_id}] Added {len(cookies_content['cookies'])} cookies, domains: {cookie_domains}")
            except Exception as e:
                logger.warning(f"[Task {task_id}] Failed to add cookies: {e}")
        
        page = await context.new_page()
        
        try:
            # 导航到起始页面
            await page.goto(target_url, wait_until='networkidle', timeout=60000)
            await asyncio.sleep(3)  # 等待页面完全加载
            
            # 创建并运行 Agent
            agent = LocalWebAgent(
                page=page,
                user_query=intent,
                output_dir=task_output_dir,
                reasoning_config=reasoning_config or REASONING_MODEL_CONFIG,
                grounder_config=grounder_config or GROUNDER_MODEL_CONFIG,
                REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST,
                original_target_url=target_url
            )
            
            result = await agent.run()
            
            # 评估部分
            score = 0.0
            configs = task_config  # 默认使用传入的 task_config
            if config_file and os.path.exists(config_file):
                try:
                    logger.info(f"[Task {task_id}] Running evaluation using config: {config_file}")
                    eval_traj = agent.get_evaluation_trajectory()
                    
                    # 读取配置文件
                    with open(config_file, 'r') as f:
                        configs = json.load(f)
                    
                    logger.info(f"[Task {task_id}] Reference answers: {configs.get('eval', {}).get('reference_answers', {})}")
                    logger.info(f"[Task {task_id}] Predict answer: {agent.final_answer}")
                    
                    # 构建符合 extract_answer 格式的 solution_str
                    formatted_solution = f"<|im_start|>assistant\n<answer>{agent.final_answer}</answer>\n<|im_end|>"
                    
                    # 创建评估器并评估
                    await asyncio.sleep(5)
                    
                    evaluator = evaluator_router(config_file, REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST)
                    score = await evaluator(
                        solution_str=formatted_solution,
                        trajectory=eval_traj,
                        config_file=config_file,
                        page=page
                    )
                    logger.info(f"[Task {task_id}] Evaluation Score: {score}")
                except Exception as e:
                    logger.error(f"[Task {task_id}] Evaluation failed: {e}")
                    traceback.print_exc()
            
            # 保存轨迹 (兼容 webarena_online_eval.py 格式)
            trajectory = agent._build_trajectory()
            
            # 将配置和分数记录到 trajectory
            trajectory.append({
                "configs": configs,
            })
            trajectory.append({
                "type": "evaluation",
                "score": float(score) if score is not None else 0.0
            })
            
            trajectory_path = os.path.join(task_output_dir, 'trajectory.json')
            with open(trajectory_path, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[Task {task_id}] Completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"[Task {task_id}] Error: {e}")
            traceback.print_exc()
            return False
            
        finally:
            await context.close()
            # 注意：不要关闭 browser，因为它是 actor 管理的
                
    except Exception as e:
        logger.error(f"[Task {task_id}] Fatal error: {e}")
        traceback.print_exc()
        return False



# =============================================================================
# Worker 函数
# =============================================================================

def worker(
    REPLACE_WITH_YOUR_HOST: str,
    browser_endpoint: str,
    task_queue: Queue,
    output_dir: str,
    webarena_auth_path: str,
    reasoning_config: Dict = None,
    grounder_config: Dict = None,
    headless: bool = True,
    dataset_path: str = None,  # 任务配置文件目录，用于评估
    reset_web: bool = False,  # 是否在开始前重置网站环境
    retest_failed: bool = False,  # 是否重测失败的任务
):
    """Worker 线程函数"""
    logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Worker started, endpoint: {browser_endpoint}")
    
    # 创建并启动 BrowserActor
    actor = BrowserActor(browser_endpoint)
    try:
        actor.start()
        logger.info(f"[{REPLACE_WITH_YOUR_HOST}] BrowserActor started")
    except Exception as e:
        logger.error(f"[{REPLACE_WITH_YOUR_HOST}] Failed to start BrowserActor: {e}")
        return
    
    while True:
        try:
            task_item = task_queue.get(timeout=5)
        except:
            break
        
        task_config = task_item['config']
        task_file = task_item.get('file', '')  # 任务配置文件名
        task_id = task_config.get('task_id', 'unknown')
        
        # 构建配置文件完整路径 (用于评估)
        config_file = None
        if dataset_path and task_file:
            config_file = os.path.join(dataset_path, task_file)
        
        logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Assigned task {task_id}")
        
        # 检查是否已完成
        sample_dir = os.path.join(output_dir, f"val_{task_id}")
        trajectory_path = os.path.join(sample_dir, "trajectory.json")
        
        if os.path.exists(trajectory_path):
            skip_task = True
            
            if retest_failed:
                # 检查是否失败
                try:
                    with open(trajectory_path, 'r', encoding='utf-8') as f:
                        traj_data = json.load(f)
                    
                    # 检查是否有评估分数
                    score = 0.0
                    has_score = False
                    for item in traj_data:
                        if item.get("type") == "evaluation":
                            score = float(item.get("score", 0.0))
                            has_score = True
                            break
                    
                    # 如果有分数且 >= 1.0 (或根据具体指标)，则视为成功
                    # 这里假设 score >= 1.0 为成功
                    if has_score and score >= 1.0:
                        logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Task {task_id} completed successfully (score: {score}), skipping")
                        skip_task = True
                    else:
                        logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Task {task_id} failed previously (score: {score}), retesting...")
                        skip_task = False
                        
                except Exception as e:
                    logger.warning(f"[{REPLACE_WITH_YOUR_HOST}] Failed to read trajectory for {task_id}, retesting... Error: {e}")
                    skip_task = False
            
            if skip_task:
                logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Task {task_id} already completed, skipping")
                task_queue.task_done()
                continue
        
        # 1. 刷新 ECS 网页状态
        try:
           
            logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Refreshing environment...")
            future = actor.submit(
                ssh_connect_and_refreshweb,
                hostname=REPLACE_WITH_YOUR_HOST,
                username="root",
                password="***",
                webarena_auth_path=f"{webarena_auth_path}_{REPLACE_WITH_YOUR_HOST}",
                owner_actor=actor
            )
            hostname, success, msg = future.result(timeout=60 * 30)
            
            if not success:
                logger.error(f"[{REPLACE_WITH_YOUR_HOST}] Refresh failed: {msg}")
                task_queue.task_done()
                continue
                
            logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Environment refreshed: {msg}")
            
        except Exception as e:
            logger.error(f"[{REPLACE_WITH_YOUR_HOST}] Refresh error: {e}")
            task_queue.task_done()
            continue
        
        # 2. 准备 URL
        start_url = task_config.get('start_url', '')
        target_url = start_url.replace("REPLACE_WITH_YOUR_HOST", REPLACE_WITH_YOUR_HOST)
        
        # 3. 加载 Cookies
        cookies_content = load_cookies_for_task(target_url, webarena_auth_path, REPLACE_WITH_YOUR_HOST)
        
        # 4. 运行本地 Agent (使用 actor.submit 提交到 Actor 的事件循环)
        # 注意：必须使用 actor.submit()，因为 browser 对象只能在创建它的事件循环中使用
        try:
            future = actor.submit(
                process_single_task_local,
                task_config=task_config,
                output_dir=output_dir,
                target_url=target_url,
                actor=actor,
                cookies_content=cookies_content,
                reasoning_config=reasoning_config,
                grounder_config=grounder_config,
                REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST,
                config_file=config_file  # 传递配置文件路径用于评估
            )
            success = future.result(timeout=TIMEOUT)
        except Exception as e:
            logger.error(f"[{REPLACE_WITH_YOUR_HOST}][Task {task_id}] Agent execution error: {e}")
            traceback.print_exc()
        
        task_queue.task_done()
    
    # 清理
    try:
        actor.stop()
    except Exception as e:
        logger.error(f"[{REPLACE_WITH_YOUR_HOST}] Error stopping actor: {e}")
    
    logger.info(f"[{REPLACE_WITH_YOUR_HOST}] Worker finished")


# =============================================================================
# 主函数
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Local WebAgent Evaluator")
    parser.add_argument("--dataset-path", type=str, 
                       default=os.environ.get("DATASET_PATH", "./config_files"),
                       help="任务配置文件目录")
    parser.add_argument("--output-dir", type=str,
                       default=os.environ.get("SAVE_MODEL_PATH", "./output"),
                       help="输出目录")
    parser.add_argument("--webarena-auth-path", type=str,
                       default=os.environ.get("WEBARENA_AUTH_PATH", "./log"),
                       help="WebArena认证路径")
    parser.add_argument("--ecs-csv", type=str,
                       default="./ecs_instances.csv",
                       help="ECS实例csv文件")
    parser.add_argument("--reset-web", action="store_true", help="是否在所有任务前重置网站环境")

    parser.add_argument("--headless", action="store_true",
                       help="使用无头浏览器")
    parser.add_argument("--num-ecs", type=int, default=5,
                       help="使用的ECS数量")
    parser.add_argument("--retest-failed", action="store_true",
                       help="是否重测失败的任务")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载 ECS IPs
    logger.info(f"Loading ECS IPs from {args.ecs_csv}...")
    REPLACE_WITH_YOUR_HOSTs = get_REPLACE_WITH_YOUR_HOSTs(args.ecs_csv)
    if not REPLACE_WITH_YOUR_HOSTs:
        logger.error("No ECS IPs found")
        return
    REPLACE_WITH_YOUR_HOSTs = REPLACE_WITH_YOUR_HOSTs[:args.num_ecs]
    logger.info(f"Using {len(REPLACE_WITH_YOUR_HOSTs)} ECS instances: {REPLACE_WITH_YOUR_HOSTs}")
    
    
    # 3. 获取浏览器 endpoints
    browser_endpoints = get_ws_endpoint_list()
    if not browser_endpoints:
        logger.error("No browser endpoints found")
        return
    logger.info(f"Found {len(browser_endpoints)} browser endpoints")
    
    # 4. 准备 worker 资源
    ###################################
    available_workers =  min(len(REPLACE_WITH_YOUR_HOSTs), len(browser_endpoints))
    ###################################
    worker_resources = []
    for i in range(available_workers):
        worker_resources.append({
            "REPLACE_WITH_YOUR_HOST": REPLACE_WITH_YOUR_HOSTs[i],
            "browser_endpoint": browser_endpoints[i]
        })
    
    # 5. 加载任务
    logger.info(f"Loading tasks from {args.dataset_path}...")
    scheduler = TaskScheduler()
    task_files = [f for f in os.listdir(args.dataset_path) if f.endswith('.json')]
    scheduler.load_tasks(args.dataset_path, task_files)
    
    all_tasks = []
    for group in scheduler.conflict_groups.values():
        all_tasks.extend(group)
    all_tasks.extend(scheduler.non_conflicting)
    
    logger.info(f"Total tasks: {len(all_tasks)}")
    
    # 6. 创建任务队列
    task_queue = Queue()
    for task in all_tasks:
        task_queue.put(task)
    
    # 7. 启动 workers
    threads = []
    for res in worker_resources:
        t = threading.Thread(
            target=worker,
            args=(
                res["REPLACE_WITH_YOUR_HOST"],
                res["browser_endpoint"],
                task_queue,
                args.output_dir,
                args.webarena_auth_path,
                REASONING_MODEL_CONFIG,
                GROUNDER_MODEL_CONFIG,
                args.headless,
                args.dataset_path,  # 传递配置文件目录用于评估
                args.reset_web,  # 是否重置网站环境
                args.retest_failed  # 是否重测失败的任务
            )
        )
        t.start()
        threads.append(t)
    
    # 8. 等待完成
    for t in threads:
        t.join()
    
    logger.info("All tasks completed!")


if __name__ == "__main__":
    main()
