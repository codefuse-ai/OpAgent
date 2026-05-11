#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local WebAgent Evaluator for OpAgent

A standalone evaluation script that runs the Reflector-Planner-Grounder-Summary
multi-agent architecture on WebArena tasks.

Features:
1. Loads ECS instances and browser endpoints
2. Runs tasks with site-specific expert tips
3. Supports any OpenAI-compatible VLM API
4. Uses OpAgent's evaluation harness for scoring

Usage:
    python eval/local_agent_eval.py \
        --dataset-path ./config_files \
        --output-dir ./output \
        --reasoning-model qwen2.5-vl-72b \
        --reasoning-base-url http://localhost:8000/v1 \
        --grounder-model qwen2.5-vl-72b \
        --grounder-base-url http://localhost:8000/v1
"""

import os
import sys
import json
import time
import asyncio
import base64
import traceback
import threading
import csv
import math
import re
from queue import Queue
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image
from loguru import logger

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import prompts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import (
    REFLECTION_PROMPT,
    PLANNER_PROMPT,
    GROUNDER_PROMPT,
    SUMMARY_PROMPT,
    get_domain_specific_tips,
    get_summary_tips,
)

# Import OpAgent modules
try:
    from opagent.evaluation_harness import evaluator_router
    from opagent.browser_env.async_envs import BrowserActor, get_ws_endpoint_list
    from opagent.browser_env.refresh_web import ssh_connect_and_refreshweb
    from opagent.task_scheduler import TaskScheduler
    from opagent.utils import extract_answer
except ImportError as e:
    logger.warning(f"Some OpAgent modules not available: {e}")
    logger.warning("Make sure opagent is installed and accessible from PYTHONPATH")

# Try import playwright
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
except ImportError:
    logger.error("playwright not installed. Run: pip install playwright && playwright install")
    raise

# =============================================================================
# Configuration
# =============================================================================

# Ensure environment variables exist
if os.environ.get('VLM_EXP_DEBUG') is None:
    os.environ['VLM_EXP_DEBUG'] = '1'
if os.environ.get('WEBARENA_AUTH_PATH') is None:
    os.environ['WEBARENA_AUTH_PATH'] = './log'

# Maximum execution steps and timeout
MAX_STEPS = int(os.environ.get('MAX_STEPS', '50'))
TIMEOUT = int(os.environ.get('TIMEOUT', str(120 * 60)))  # 2 hours

# Site port mapping
SITE_PORT_MAP = {
    "7770": "shopping",
    "7780": "shopping_admin",
    "9999": "reddit",
    "8023": "gitlab",
    "8888": "wikipedia",
    "3000": "map",
    "4399": "homepage",
}

# Default model configs (override via CLI args or env vars)
REASONING_MODEL_CONFIG = {
    "model": os.environ.get('REASONING_MODEL', 'qwen2.5-vl-72b'),
    "base_url": os.environ.get('REASONING_BASE_URL', 'http://localhost:8000/v1'),
    "api_key": os.environ.get('REASONING_API_KEY', 'EMPTY'),
    "temperature": 0.0,
}

GROUNDER_MODEL_CONFIG = {
    "model": os.environ.get('GROUNDER_MODEL', 'qwen2.5-vl-72b'),
    "base_url": os.environ.get('GROUNDER_BASE_URL', 'http://localhost:8000/v1'),
    "api_key": os.environ.get('GROUNDER_API_KEY', 'EMPTY'),
    "temperature": 0.0,
}

# WebArena host
WEBHOSTNAME = os.environ.get('WEBHOSTNAME', 'http://localhost')

# ECS SSH credentials (set via environment variables)
ECS_SSH_USERNAME = os.environ.get('ECS_SSH_USERNAME', 'root')
ECS_SSH_PASSWORD = os.environ.get('ECS_SSH_PASSWORD', '')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StepInfo:
    step_id: int
    action: Dict[str, Any]
    screenshot_path: str = ""
    screenshot_base64: str = ""
    page_url: str = ""
    error_msg: str = ""
    reflection: Dict[str, Any] = field(default_factory=dict)
    planner_output: Dict[str, Any] = field(default_factory=dict)
    grounder_output: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


# =============================================================================
# Model Caller
# =============================================================================

class ModelCaller:
    """Generic OpenAI-compatible VLM API caller."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:8000/v1')
        self.api_key = config.get('api_key', 'EMPTY')
        self.model = config.get('model', 'qwen2.5-vl-72b')
        self.temperature = config.get('temperature', 0.0)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse model response into structured dict."""
        try:
            import json_repair
        except ImportError:
            json_repair = None

        # 1. Try parsing ◁tool_call▷ format (Grounder SFT model output)
        tool_call_match = re.search(r'◁tool_call▷(.*?)◁/tool_call▷', response_text, re.DOTALL)
        if tool_call_match:
            try:
                tool_call_str = tool_call_match.group(1).strip()
                if json_repair:
                    return json_repair.loads(tool_call_str)
                return json.loads(tool_call_str)
            except Exception:
                pass

        # 2. Try parsing <answer> format
        answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if answer_match:
            return {"answer": answer_match.group(1).strip(), "is_answer": True}

        # 3. Try parsing JSON code block
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                if json_repair:
                    return json_repair.loads(json_str)
                return json.loads(json_str)
            except Exception:
                pass

        return {"raw_response": response_text}

    async def call(self, prompt: str, image_list: List[str] = None,
                   max_retries: int = 5) -> Dict[str, Any]:
        """Call VLM model via OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.error("openai not installed. Run: pip install openai")
            return {}

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Build message content
        content = []

        # Add images first (if any)
        if image_list:
            for img in image_list:
                if img and img != '' and img != 'None':
                    if img.startswith('http'):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img},
                        })
                    elif img.startswith('data:'):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img},
                        })
                    else:
                        # Assume base64
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        })

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": "You are an advanced web agent."},
            {"role": "user", "content": content}
        ]

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )

                if not response or not response.choices:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    continue

                response_text = response.choices[0].message.content
                if not response_text:
                    logger.warning(f"Empty content on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    continue

                return self._parse_response(response_text)

            except Exception as e:
                logger.error(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)

        return {}


# =============================================================================
# Utility Functions
# =============================================================================

def resize_image_base64(base64_str: str, max_size: int = 1024) -> str:
    """Resize image and return base64 string."""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to resize image: {e}")
        return base64_str


async def screenshot_to_base64(page: Page) -> str:
    """Take a screenshot and return base64 string."""
    screenshot_bytes = await page.screenshot(full_page=False)
    return base64.b64encode(screenshot_bytes).decode('utf-8')


async def save_screenshot(page: Page, save_path: str) -> str:
    """Save screenshot to file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    await page.screenshot(path=save_path, full_page=False)
    return save_path


async def execute_browser_action(page: Page, action_type: str, params: Dict[str, Any]) -> str:
    """Execute a browser action and return error message (empty string if success)."""
    try:
        if action_type == 'click':
            coords = params.get('coords', [])
            if coords and len(coords) >= 2:
                x, y = int(coords[0]), int(coords[1])
                await page.mouse.click(x, y)
            else:
                return "No coordinates for click action"

        elif action_type == 'type':
            coords = params.get('coords', [])
            content = params.get('content', '')
            press_enter = params.get('press_enter', True)
            if coords and len(coords) >= 2:
                x, y = int(coords[0]), int(coords[1])
                await page.mouse.click(x, y)
                await asyncio.sleep(0.3)
                await page.keyboard.type(content, delay=50)
                if press_enter:
                    await page.keyboard.press('Enter')
            else:
                return "No coordinates for type action"

        elif action_type == 'hover':
            coords = params.get('coords', [])
            if coords and len(coords) >= 2:
                x, y = int(coords[0]), int(coords[1])
                await page.mouse.move(x, y)

        elif action_type == 'scroll':
            coords = params.get('coords', [])
            direction = params.get('direction', 'down')
            if coords and len(coords) >= 2:
                x, y = int(coords[0]), int(coords[1])
                delta_y = 500 if direction == 'down' else -500 if direction == 'up' else 0
                delta_x = 500 if direction == 'right' else -500 if direction == 'left' else 0
                await page.mouse.wheel(delta_x, delta_y)
            else:
                delta_y = 500 if direction == 'down' else -500 if direction == 'up' else 0
                delta_x = 500 if direction == 'right' else -500 if direction == 'left' else 0
                await page.evaluate(f"window.scrollBy({delta_x}, {delta_y})")

        elif action_type == 'goto':
            url = params.get('url', '')
            if url:
                await page.goto(url, wait_until='networkidle', timeout=60000)
            else:
                return "No URL for goto action"

        elif action_type == 'go_back':
            await page.go_back(wait_until='networkidle', timeout=60000)

        elif action_type == 'go_forward':
            await page.go_forward(wait_until='networkidle', timeout=60000)

        elif action_type == 'press':
            key = params.get('key', 'Enter')
            await page.keyboard.press(key)

        elif action_type == 'select_option':
            coords = params.get('coords', [])
            option = params.get('option', '')
            if coords and len(coords) >= 2:
                x, y = int(coords[0]), int(coords[1])
                await page.mouse.click(x, y)
                await asyncio.sleep(0.5)
            if option:
                # Try using Playwright's select_option
                try:
                    select_el = await page.query_selector('select:focus')
                    if select_el:
                        await select_el.select_option(label=option)
                    else:
                        await page.keyboard.type(option)
                        await page.keyboard.press('Enter')
                except Exception:
                    await page.keyboard.type(option)
                    await page.keyboard.press('Enter')

        elif action_type == 'right_click':
            coords = params.get('coords', [])
            if coords and len(coords) >= 2:
                x, y = int(coords[0]), int(coords[1])
                await page.mouse.click(x, y, button='right')

        elif action_type == 'zoom_in':
            level = params.get('level', 1)
            for _ in range(level):
                await page.keyboard.press('Control+Equal')

        elif action_type == 'zoom_out':
            level = params.get('level', 1)
            for _ in range(level):
                await page.keyboard.press('Control+Minus')

        else:
            return f"Unknown action type: {action_type}"

        # Wait for page to stabilize
        await asyncio.sleep(1)
        try:
            await page.wait_for_load_state('networkidle', timeout=5000)
        except Exception:
            pass

        return ""

    except Exception as e:
        return str(e)


def get_ecs_ips(file_path: str) -> List[str]:
    """Read ECS IP list from CSV file."""
    ips = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                idx = header.index("公网 IP")
            except ValueError:
                idx = 1
            for row in reader:
                if len(row) > idx and row[idx].strip():
                    ips.append(row[idx].strip())
    except Exception as e:
        logger.error(f"Error reading ECS IPs: {e}")
    return ips


def load_cookies_for_task(target_url: str, webarena_auth_path: str, ecs_ip: str) -> Optional[Dict]:
    """Load cookies based on task URL."""
    auth_dir = f"{webarena_auth_path}_{ecs_ip}/.auth"

    sites_needed = set()
    for port, site_name in SITE_PORT_MAP.items():
        if f":{port}" in target_url:
            sites_needed.add(site_name)

    if "shopping" in sites_needed or "shopping_admin" in sites_needed:
        sites_needed.update(["shopping", "shopping_admin"])
    if "gitlab" in sites_needed or "reddit" in sites_needed:
        sites_needed.update(["gitlab", "reddit"])

    if not sites_needed:
        return None

    cookie_files_to_try = []
    sorted_sites = sorted(list(sites_needed))
    cookie_files_to_try.append(f"{'.'.join(sorted_sites)}_state.json")
    for site in sorted_sites:
        cookie_files_to_try.append(f"{site}_state.json")

    for cookie_filename in cookie_files_to_try:
        cookie_file_path = os.path.join(auth_dir, cookie_filename)
        try:
            if os.path.exists(cookie_file_path):
                with open(cookie_file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "cookies" in data and len(data["cookies"]) > 0:
                        logger.info(f"[{ecs_ip}] Loaded {len(data['cookies'])} cookies from {cookie_file_path}")
                        return data
        except Exception as e:
            logger.warning(f"[{ecs_ip}] Failed to read {cookie_filename}: {e}")

    logger.warning(f"[{ecs_ip}] Cookie file not found for {target_url}")
    return None


# =============================================================================
# LocalWebAgent
# =============================================================================

class LocalWebAgent:
    """Multi-agent web automation system: Reflector -> Planner -> Grounder -> Action."""

    def __init__(
        self,
        page: Page,
        user_query: str,
        output_dir: str,
        reasoning_config: Dict = None,
        grounder_config: Dict = None,
        host: str = None,
        original_target_url: str = "",
    ):
        self.page = page
        self.user_query = user_query
        self.output_dir = output_dir
        self.host = host or WEBHOSTNAME.replace('http://', '').replace('https://', '')
        self.original_target_url = original_target_url

        self.reasoning_caller = ModelCaller(reasoning_config or REASONING_MODEL_CONFIG)
        self.grounder_caller = ModelCaller(grounder_config or GROUNDER_MODEL_CONFIG)

        self.steps: List[StepInfo] = []
        self.marked_notes: List[str] = []
        self.todo_list: List[Dict] = []
        self.current_step = 0
        self.max_steps = MAX_STEPS
        self.timeout = TIMEOUT
        self.is_finished = False
        self.final_answer = ""
        self.last_screenshot = ""

        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # -----------------------------------------------------------------
    # Context helpers
    # -----------------------------------------------------------------

    def get_recent_steps_detail(self, n: int = 3) -> str:
        if not self.steps:
            return "No execution history"
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
        if not self.steps:
            return "No execution path"
        path_items = []
        for step in self.steps:
            action = step.action
            action_type = action.get('action_type', 'unknown')
            path_items.append(f"Step {step.step_id}: {action_type}")
        return " -> ".join(path_items)

    def get_marked_notes(self) -> str:
        if not self.marked_notes:
            return "None"
        return "\n".join([f"{i+1}. {note}" for i, note in enumerate(self.marked_notes)])

    def get_todolist_status(self) -> str:
        if not self.todo_list:
            return "None"
        items = []
        for item in self.todo_list:
            status_icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳", "failed": "❌"}.get(item.get('status', ''), '⏳')
            items.append(f"{status_icon} [{item.get('id', '')}] {item.get('description', '')} ({item.get('status', '')})")
        return "\n".join(items)

    # -----------------------------------------------------------------
    # Module calls
    # -----------------------------------------------------------------

    async def call_reflector(self, prev_screenshot: str, curr_screenshot: str,
                              error_msg: str = "") -> Dict[str, Any]:
        domain_tips = get_domain_specific_tips(self.page.url, self.host)

        prompt = REFLECTION_PROMPT.format(
            user_query=self.user_query,
            html_simplify="None",
            recent_3_step_details=self.get_recent_steps_detail(3),
            marked_note=self.get_marked_notes(),
            current_path=self.get_execution_path(),
            todolist_status=self.get_todolist_status(),
            tips=domain_tips,
        )

        images = []
        if prev_screenshot:
            images.append(prev_screenshot)
        if curr_screenshot:
            images.append(curr_screenshot)
        if not images and curr_screenshot:
            images = [curr_screenshot]

        result = await self.reasoning_caller.call(prompt, images)
        logger.debug(f"[Reflector] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    async def call_planner(self, screenshot: str, reflection_signal: str = "") -> Dict[str, Any]:
        domain_tips = get_domain_specific_tips(self.page.url, self.host, self.original_target_url)

        prompt = PLANNER_PROMPT.format(
            user_query=self.user_query,
            tips=domain_tips,
            todo_list_status=self.get_todolist_status(),
            reflection_signal=reflection_signal or "None",
            marked_note=self.get_marked_notes(),
            recent_3_step_details=self.get_recent_steps_detail(3),
        )

        result = await self.reasoning_caller.call(prompt, [screenshot])
        logger.debug(f"[Planner] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    async def call_grounder(self, screenshot: str, instruction: str) -> Dict[str, Any]:
        description = f"Your overall goal is: {self.user_query}; Current goal is: {instruction}"
        prompt = GROUNDER_PROMPT.format(instruction=description)

        # Add history actions
        if self.steps:
            prompt += "Previous actions:\n"
            for step in self.steps[-2:]:
                action_str = f"Step {step.step_id}:\n<image>\n{json.dumps(step.action, ensure_ascii=False)}"
                prompt += action_str + "\n\n"

        # Build image list: current + up to 2 historical
        images = [screenshot]
        for step in self.steps[-2:]:
            if step.screenshot_base64:
                images.append(step.screenshot_base64)

        result = await self.grounder_caller.call(prompt, images)

        if result is None:
            result = {}

        # Extract coords
        coords = None
        if 'coords' in result:
            coords = result['coords']
        elif 'arguments' in result and 'coords' in result.get('arguments', {}):
            coords = result['arguments']['coords']

        if coords and len(coords) >= 2:
            if 'coords' in result:
                result['coords'] = [int(c) for c in coords]
            elif 'arguments' in result and 'coords' in result.get('arguments', {}):
                result['arguments']['coords'] = [int(c) for c in coords]

        logger.debug(f"[Grounder] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    async def call_summary(self, current_screenshot: str, max_screenshots: int = 5) -> Dict[str, Any]:
        execution_history = self.get_recent_steps_detail(10)

        screenshots = []
        recent_steps = self.steps[-max_screenshots:] if len(self.steps) >= max_screenshots else self.steps
        for step in recent_steps:
            if step.screenshot_base64:
                screenshots.append(step.screenshot_base64)
        if current_screenshot:
            screenshots.append(current_screenshot)
        if not screenshots and current_screenshot:
            screenshots = [current_screenshot]
        if len(screenshots) > max_screenshots:
            screenshots = screenshots[-max_screenshots:]

        summary_tips = get_summary_tips(self.page.url, self.host)
        marked_note = self.get_marked_notes()

        prompt = SUMMARY_PROMPT.format(
            user_query=self.user_query,
            tips=summary_tips,
            execution_history=execution_history,
            marked_note=marked_note,
            screenshot_count=len(screenshots),
        )

        logger.info(f"[Summary] Using {len(screenshots)} screenshots")
        result = await self.reasoning_caller.call(prompt, screenshots)
        logger.debug(f"[Summary] Result: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        start_time = time.time()

        prev_screenshot = ""
        curr_screenshot = await screenshot_to_base64(self.page)
        self.last_screenshot = curr_screenshot
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
                # 1. Reflector
                reflection_result = await self.call_reflector(
                    prev_screenshot, curr_screenshot,
                    self.steps[-1].error_msg if self.steps else ""
                )

                if reflection_result.get('is_task_done', False):
                    logger.info("[Agent] Task marked as done by Reflector")
                    summary_result = await self.call_summary(curr_screenshot)
                    self.final_answer = summary_result.get('answer', '')
                    self.is_finished = True
                    break

                # Update todo_list
                if reflection_result.get('todo_list'):
                    self.todo_list = reflection_result['todo_list']
                    logger.info(f"[Agent] Todo list updated: {len(self.todo_list)} items")

                # Update marked notes
                if reflection_result.get('mark_node', False) and reflection_result.get('note'):
                    self.marked_notes.append(reflection_result['note'])

                # 2. Planner
                reflection_signal = ""
                if reflection_result.get('instruction'):
                    inst = reflection_result['instruction']
                    if isinstance(inst, dict):
                        reflection_signal = f"Level: {inst.get('level', '')}, Content: {inst.get('content', '')}"
                    else:
                        reflection_signal = str(inst)

                planner_result = await self.call_planner(curr_screenshot, reflection_signal)

                action_type = planner_result.get('action_type', '')
                action_attrs = planner_result.get('action_attributes', {}) or {}
                instruction = planner_result.get('instruction', '')

                if not action_type:
                    logger.warning("[Agent] Planner returned no action")
                    continue

                # Check for stop action
                if action_type.lower() in ['stop', 'finish', 'done', 'answer']:
                    logger.info("[Agent] Planner returned stop action")
                    summary_result = await self.call_summary(curr_screenshot)
                    self.final_answer = summary_result.get('answer', action_attrs.get('answer', ''))
                    self.is_finished = True
                    break

                # 3. Grounder (for coordinate-based actions)
                grounder_result = {}
                if action_type in ['click', 'type', 'hover', 'scroll', 'select_option',
                                   'right_click', 'zoom_in', 'zoom_out'] and instruction:
                    grounder_result = await self.call_grounder(curr_screenshot, instruction)

                    if 'coords' in grounder_result:
                        action_attrs['coords'] = grounder_result['coords']
                    elif 'arguments' in grounder_result and 'coords' in grounder_result.get('arguments', {}):
                        action_attrs['coords'] = grounder_result['arguments']['coords']

                    if action_type == 'type':
                        if 'content' not in action_attrs and 'text' in grounder_result:
                            action_attrs['content'] = grounder_result['text']
                        if 'content' not in action_attrs and 'arguments' in grounder_result:
                            action_attrs['content'] = grounder_result['arguments'].get('text', '')

                # Build final action
                action = {
                    'action_type': action_type,
                    'params': action_attrs
                }

                # 4. Execute action
                logger.info(f"[Agent] Executing: {action_type} with {action_attrs}")
                error_msg = await execute_browser_action(self.page, action_type, action_attrs)

                # 5. Record step
                step_screenshot_base64 = curr_screenshot
                prev_screenshot = curr_screenshot
                curr_screenshot = await screenshot_to_base64(self.page)
                self.last_screenshot = curr_screenshot
                screenshot_path = await save_screenshot(
                    self.page,
                    os.path.join(self.output_dir, "images", f"screenshot_{self.current_step}.png")
                )

                step_info = StepInfo(
                    step_id=self.current_step,
                    action=action,
                    screenshot_path=screenshot_path,
                    screenshot_base64=step_screenshot_base64,
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

        # Generate final result if not finished
        if not self.is_finished:
            try:
                logger.info("[Agent] Generating final answer...")
                summary_result = await self.call_summary(curr_screenshot)
                answer = summary_result.get('answer', '')
                if answer:
                    self.final_answer = answer
            except Exception as e:
                logger.error(f"[Agent] Summary failed: {e}")

        # Save trajectory
        trajectory = self._build_trajectory()
        trajectory_path = os.path.join(self.output_dir, 'trajectory.json')
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        logger.info(f"[Agent] Task completed. Steps: {self.current_step}, "
                     f"Answer: {self.final_answer[:100] if self.final_answer else 'N/A'}")

        return {
            'trajectory': trajectory,
            'final_answer': self.final_answer,
            'is_finished': self.is_finished,
            'total_steps': self.current_step,
            'steps': [asdict(s) for s in self.steps]
        }

    def _build_trajectory(self) -> List[Dict]:
        trajectory = []
        trajectory.append({"type": "observation", "image_path": "images/screenshot_0.png"})

        for step in self.steps:
            trajectory.append({
                "type": "action",
                "action": step.action,
                "reflector_output": step.reflection,
                "planner_output": step.planner_output,
                "grounder_output": step.grounder_output,
                "page_url": step.page_url,
                "error_msg": step.error_msg
            })
            trajectory.append({
                "type": "observation",
                "image_path": f"images/screenshot_{step.step_id}.png"
            })

        if self.final_answer:
            trajectory.append({
                "type": "generated_answer",
                "generated_final_answer": f"<|im_start|>assistant\n<answer>{self.final_answer}</answer>\n<|im_end|>"
            })

        return trajectory

    def get_evaluation_trajectory(self) -> List[Any]:
        traj = []
        for step in self.steps:
            if step.screenshot_base64:
                try:
                    img_data = base64.b64decode(step.screenshot_base64)
                    image = Image.open(BytesIO(img_data))
                    observation = {"image": np.array(image), "text": ""}
                    state_info = {"observation": observation, "info": {"page": None, "url": step.page_url}}
                    traj.append(state_info)
                except Exception as e:
                    logger.warning(f"Failed to decode image for step {step.step_id}: {e}")

            action_data = step.action
            flat_action = {"action_type": action_data.get('action_type')}
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
# Task Processing
# =============================================================================

async def process_single_task(
    task_config: Dict[str, Any],
    output_dir: str,
    target_url: str,
    actor: BrowserActor,
    cookies_content: Dict = None,
    reasoning_config: Dict = None,
    grounder_config: Dict = None,
    host: str = None,
    config_file: str = None,
) -> bool:
    task_id = task_config.get('task_id', 'unknown')
    intent = task_config.get('intent', '')

    task_output_dir = os.path.join(output_dir, f"val_{task_id}")
    os.makedirs(task_output_dir, exist_ok=True)

    logger.info(f"[Task {task_id}] Starting on {target_url[:50]}...")

    try:
        browser = actor.browser_unit.browser

        context = await browser.new_context(
            viewport={"width": 1440, "height": 1200},
            ignore_https_errors=True,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        )

        if cookies_content and "cookies" in cookies_content:
            try:
                await context.add_cookies(cookies_content["cookies"])
                logger.info(f"[Task {task_id}] Added {len(cookies_content['cookies'])} cookies")
            except Exception as e:
                logger.warning(f"[Task {task_id}] Failed to add cookies: {e}")

        page = await context.new_page()

        try:
            await page.goto(target_url, wait_until='networkidle', timeout=60000)
            await asyncio.sleep(3)

            agent = LocalWebAgent(
                page=page,
                user_query=intent,
                output_dir=task_output_dir,
                reasoning_config=reasoning_config or REASONING_MODEL_CONFIG,
                grounder_config=grounder_config or GROUNDER_MODEL_CONFIG,
                host=host,
                original_target_url=target_url,
            )

            result = await agent.run()

            # Evaluation
            score = 0.0
            configs = task_config
            if config_file and os.path.exists(config_file):
                try:
                    logger.info(f"[Task {task_id}] Running evaluation using config: {config_file}")
                    eval_traj = agent.get_evaluation_trajectory()

                    with open(config_file, 'r') as f:
                        configs = json.load(f)

                    logger.info(f"[Task {task_id}] Reference answers: "
                                f"{configs.get('eval', {}).get('reference_answers', {})}")
                    logger.info(f"[Task {task_id}] Predict answer: {agent.final_answer}")

                    formatted_solution = f"<|im_start|>assistant\n<answer>{agent.final_answer}</answer>\n<|im_end|>"

                    await asyncio.sleep(5)

                    try:
                        evaluator = evaluator_router(config_file, REPLACE_WITH_YOUR_HOST=host)
                        score = await evaluator(
                            solution_str=formatted_solution,
                            trajectory=eval_traj,
                            config_file=config_file,
                            page=page
                        )
                    except TypeError:
                        # Fallback: evaluator_router may not accept REPLACE_WITH_YOUR_HOST
                        evaluator = evaluator_router(config_file)
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

            # Save trajectory with score
            trajectory = agent._build_trajectory()
            trajectory.append({"configs": configs})
            trajectory.append({"type": "evaluation", "score": float(score) if score is not None else 0.0})

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

    except Exception as e:
        logger.error(f"[Task {task_id}] Fatal error: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# Worker
# =============================================================================

def worker(
    ecs_ip: str,
    browser_endpoint: str,
    task_queue: Queue,
    output_dir: str,
    webarena_auth_path: str,
    reasoning_config: Dict = None,
    grounder_config: Dict = None,
    headless: bool = True,
    dataset_path: str = None,
    reset_web: bool = False,
    host: str = None,
):
    logger.info(f"[{ecs_ip}] Worker started, endpoint: {browser_endpoint}")

    actor = BrowserActor(browser_endpoint)
    try:
        actor.start()
        logger.info(f"[{ecs_ip}] BrowserActor started")
    except Exception as e:
        logger.error(f"[{ecs_ip}] Failed to start BrowserActor: {e}")
        return

    # Reset web environment once before starting
    if reset_web:
        logger.info(f"[{ecs_ip}] Resetting web environment...")
        try:
            future = actor.submit(
                ssh_connect_and_refreshweb,
                hostname=ecs_ip,
                username=ECS_SSH_USERNAME,
                password=ECS_SSH_PASSWORD,
                webarena_auth_path=f"{webarena_auth_path}_{ecs_ip}",
                owner_actor=actor
            )
            hostname, success, msg = future.result(timeout=600)
            if success:
                logger.info(f"[{ecs_ip}] Web reset completed: {msg}")
            else:
                logger.error(f"[{ecs_ip}] Web reset failed: {msg}")
        except Exception as e:
            logger.error(f"[{ecs_ip}] Web reset error: {e}")

    while True:
        try:
            task_item = task_queue.get(timeout=5)
        except Exception:
            break

        task_config = task_item['config']
        task_file = task_item.get('file', '')
        task_id = task_config.get('task_id', 'unknown')

        config_file = None
        if dataset_path and task_file:
            config_file = os.path.join(dataset_path, task_file)

        logger.info(f"[{ecs_ip}] Assigned task {task_id}")

        # Skip if already completed
        sample_dir = os.path.join(output_dir, f"val_{task_id}")
        trajectory_path = os.path.join(sample_dir, "trajectory.json")
        if os.path.exists(trajectory_path):
            logger.info(f"[{ecs_ip}] Task {task_id} already completed, skipping")
            task_queue.task_done()
            continue

        # Refresh ECS environment
        try:
            logger.info(f"[{ecs_ip}] Refreshing environment...")
            future = actor.submit(
                ssh_connect_and_refreshweb,
                hostname=ecs_ip,
                username=ECS_SSH_USERNAME,
                password=ECS_SSH_PASSWORD,
                webarena_auth_path=f"{webarena_auth_path}_{ecs_ip}",
                owner_actor=actor
            )
            hostname, success, msg = future.result(timeout=600)
            if not success:
                logger.error(f"[{ecs_ip}] Refresh failed: {msg}")
                task_queue.task_done()
                continue
            logger.info(f"[{ecs_ip}] Environment refreshed: {msg}")
        except Exception as e:
            logger.error(f"[{ecs_ip}] Refresh error: {e}")
            task_queue.task_done()
            continue

        # Prepare URL
        start_url = task_config.get('start_url', '')
        target_url = start_url.replace("REPLACE_WITH_YOUR_HOST", ecs_ip)

        # Load cookies
        cookies_content = load_cookies_for_task(target_url, webarena_auth_path, ecs_ip)

        # Run agent
        try:
            future = actor.submit(
                process_single_task,
                task_config=task_config,
                output_dir=output_dir,
                target_url=target_url,
                actor=actor,
                cookies_content=cookies_content,
                reasoning_config=reasoning_config,
                grounder_config=grounder_config,
                host=ecs_ip,
                config_file=config_file,
            )
            success = future.result(timeout=TIMEOUT)
        except Exception as e:
            logger.error(f"[{ecs_ip}][Task {task_id}] Agent execution error: {e}")
            traceback.print_exc()

        task_queue.task_done()

    try:
        actor.stop()
    except Exception:
        pass

    logger.info(f"[{ecs_ip}] Worker finished")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpAgent Local WebAgent Evaluator")
    parser.add_argument("--dataset-path", type=str,
                        default=os.environ.get("DATASET_PATH", "./config_files"),
                        help="Task config directory")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SAVE_MODEL_PATH", "./output"),
                        help="Output directory")
    parser.add_argument("--webarena-auth-path", type=str,
                        default=os.environ.get("WEBARENA_AUTH_PATH", "./log"),
                        help="WebArena auth path")
    parser.add_argument("--ecs-csv", type=str,
                        default="./ecs_instances.csv",
                        help="ECS instances CSV file")
    parser.add_argument("--reset-web", action="store_true",
                        help="Reset web environment before tasks")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless browser")
    parser.add_argument("--num-ecs", type=int, default=5,
                        help="Number of ECS instances to use")
    parser.add_argument("--reasoning-model", type=str, default=None,
                        help="Reasoning model name (overrides REASONING_MODEL env)")
    parser.add_argument("--reasoning-base-url", type=str, default=None,
                        help="Reasoning model base URL (overrides REASONING_BASE_URL env)")
    parser.add_argument("--reasoning-api-key", type=str, default=None,
                        help="Reasoning model API key")
    parser.add_argument("--grounder-model", type=str, default=None,
                        help="Grounder model name (overrides GROUNDER_MODEL env)")
    parser.add_argument("--grounder-base-url", type=str, default=None,
                        help="Grounder model base URL (overrides GROUNDER_BASE_URL env)")
    parser.add_argument("--grounder-api-key", type=str, default=None,
                        help="Grounder model API key")
    parser.add_argument("--webhostname", type=str, default=None,
                        help="WebArena host URL (overrides WEBHOSTNAME env)")

    args = parser.parse_args()

    # Override model configs from CLI args
    if args.reasoning_model:
        REASONING_MODEL_CONFIG['model'] = args.reasoning_model
    if args.reasoning_base_url:
        REASONING_MODEL_CONFIG['base_url'] = args.reasoning_base_url
    if args.reasoning_api_key:
        REASONING_MODEL_CONFIG['api_key'] = args.reasoning_api_key
    if args.grounder_model:
        GROUNDER_MODEL_CONFIG['model'] = args.grounder_model
    if args.grounder_base_url:
        GROUNDER_MODEL_CONFIG['base_url'] = args.grounder_base_url
    if args.grounder_api_key:
        GROUNDER_MODEL_CONFIG['api_key'] = args.grounder_api_key
    if args.webhostname:
        global WEBHOSTNAME
        WEBHOSTNAME = args.webhostname

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load ECS IPs
    logger.info(f"Loading ECS IPs from {args.ecs_csv}...")
    ecs_ips = get_ecs_ips(args.ecs_csv)
    if not ecs_ips:
        logger.error("No ECS IPs found")
        return
    ecs_ips = ecs_ips[:args.num_ecs]
    logger.info(f"Using {len(ecs_ips)} ECS instances: {ecs_ips}")

    # 2. Get browser endpoints
    browser_endpoints = get_ws_endpoint_list()
    if not browser_endpoints:
        logger.error("No browser endpoints found. Make sure browsers are initialized.")
        return
    logger.info(f"Found {len(browser_endpoints)} browser endpoints")

    # 3. Prepare worker resources
    available_workers = min(len(ecs_ips), len(browser_endpoints))
    worker_resources = []
    for i in range(available_workers):
        worker_resources.append({
            "ecs_ip": ecs_ips[i],
            "browser_endpoint": browser_endpoints[i]
        })

    # 4. Load tasks
    logger.info(f"Loading tasks from {args.dataset_path}...")
    scheduler = TaskScheduler()
    task_files = [f for f in os.listdir(args.dataset_path) if f.endswith('.json')]
    scheduler.load_tasks(args.dataset_path, task_files)

    all_tasks = []
    for group in scheduler.conflict_groups.values():
        all_tasks.extend(group)
    all_tasks.extend(scheduler.non_conflicting)

    logger.info(f"Total tasks: {len(all_tasks)}")

    # 5. Create task queue
    task_queue = Queue()
    for task in all_tasks:
        task_queue.put(task)

    # 6. Start workers
    threads = []
    for res in worker_resources:
        t = threading.Thread(
            target=worker,
            args=(
                res["ecs_ip"],
                res["browser_endpoint"],
                task_queue,
                args.output_dir,
                args.webarena_auth_path,
                REASONING_MODEL_CONFIG,
                GROUNDER_MODEL_CONFIG,
                args.headless,
                args.dataset_path,
                args.reset_web,
                res["ecs_ip"],
            )
        )
        t.start()
        threads.append(t)

    # 7. Wait for completion
    for t in threads:
        t.join()

    logger.info("All tasks completed!")


if __name__ == "__main__":
    main()