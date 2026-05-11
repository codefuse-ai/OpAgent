#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gradio frontend for remotely controlling an ECS-hosted OpAgent worker.

Environment variables:
    OPAGENT_API_BASE_URL   ECS API base URL, e.g. https://your-ecs-host:8000
    OPAGENT_API_KEY        Optional API key sent as X-API-Key
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
import time
from io import BytesIO
from typing import Any, Dict, Generator, Optional, Tuple

import gradio as gr
import requests
from PIL import Image


DEFAULT_API_BASE_URL = os.getenv("OPAGENT_API_BASE_URL", "http://218.244.136.240:8000").rstrip("/")
DEFAULT_API_KEY = os.getenv("OPAGENT_API_KEY", "opagent_demo_2026_x8Kp9LmQ2vR7")
DEFAULT_POLL_INTERVAL = float(os.getenv("OPAGENT_POLL_INTERVAL", "5"))
DEFAULT_TIMEOUT = float(os.getenv("OPAGENT_HTTP_TIMEOUT", "300"))
DEFAULT_SUBMIT_TIMEOUT = float(os.getenv("OPAGENT_SUBMIT_TIMEOUT", str(DEFAULT_TIMEOUT)))
DEFAULT_STATUS_TIMEOUT = float(os.getenv("OPAGENT_STATUS_TIMEOUT", str(DEFAULT_TIMEOUT)))
DEFAULT_IMAGE_TIMEOUT = float(os.getenv("OPAGENT_IMAGE_TIMEOUT", str(DEFAULT_TIMEOUT)))
DEFAULT_START_URL = os.getenv("OPAGENT_DEFAULT_START_URL", "https://www.bilibili.com")
DEFAULT_QUERY = os.getenv("OPAGENT_DEFAULT_QUERY", "帮我打开一个黑神话的实机视频")
USE_SYSTEM_PROXY = os.getenv("OPAGENT_USE_SYSTEM_PROXY", "0") == "1"


@lru_cache(maxsize=1)
def get_http_session() -> requests.Session:
    session = requests.Session()
    if not USE_SYSTEM_PROXY:
        session.trust_env = False
        session.proxies.clear()
    return session


def build_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    return headers


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("API base URL cannot be empty")
    return base_url


def fetch_image(image_url: Optional[str], api_key: str) -> Optional[Image.Image]:
    if not image_url:
        return None

    response = get_http_session().get(
        image_url,
        headers={"X-API-Key": api_key.strip()} if api_key.strip() else None,
        timeout=DEFAULT_IMAGE_TIMEOUT,
    )
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def format_status(task: Dict[str, Any]) -> str:
    lines = [
        f"### Task Status: {task.get('status', 'unknown')}",
        f"- Current Step: {task.get('current_step', 0)} / {task.get('max_steps', 0)}",
        f"- Current Page: {task.get('current_url') or '-'}",
    ]

    latest_action = task.get("latest_action")
    if latest_action:
        lines.append("- Latest Action:")
        lines.append(f"\n```json\n{json.dumps(latest_action, ensure_ascii=False)}\n```")

    latest_think = task.get("latest_think")
    if latest_think:
        preview = latest_think[:500] + ("..." if len(latest_think) > 500 else "")
        lines.append("- Latest Reasoning:")
        lines.append(f"\n> {preview}")

    final_answer = task.get("final_answer")
    if final_answer:
        lines.append("\n### Final Answer")
        lines.append(final_answer)

    error = task.get("error")
    if error:
        lines.append("\n### Error")
        lines.append(error)

    events = task.get("events") or []
    if events:
        lines.append("\n### Recent Logs")
        lines.extend([f"- {event}" for event in events[-12:]])

    return "\n".join(lines)


def format_raw_response(task: Dict[str, Any]) -> str:
    raw_response = task.get("latest_raw_response") or ""
    if not raw_response:
        return ""
    return raw_response[:12000]


def submit_task(
    api_base_url: str,
    api_key: str,
    start_url: str,
    query: str,
    max_steps: int,
) -> Dict[str, Any]:
    payload = {
        "url": start_url.strip() or None,
        "query": query.strip(),
        "max_steps": int(max_steps),
    }
    response = get_http_session().post(
        f"{normalize_base_url(api_base_url)}/tasks",
        headers=build_headers(api_key),
        json=payload,
        timeout=DEFAULT_SUBMIT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def get_task_status(api_base_url: str, api_key: str, task_id: str) -> Dict[str, Any]:
    response = get_http_session().get(
        f"{normalize_base_url(api_base_url)}/tasks/{task_id}",
        headers=build_headers(api_key),
        timeout=DEFAULT_STATUS_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def cancel_remote_task(api_base_url: str, api_key: str, raw_status: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    task_id = (raw_status or {}).get("task_id")
    if not task_id:
        raise gr.Error("There is no running task to cancel")

    response = get_http_session().post(
        f"{normalize_base_url(api_base_url)}/tasks/{task_id}/cancel",
        headers=build_headers(api_key),
        timeout=DEFAULT_STATUS_TIMEOUT,
    )
    response.raise_for_status()
    cancel_result = response.json()

    task = get_task_status(api_base_url, api_key, task_id)
    task.setdefault("events", [])
    if cancel_result.get("message"):
        task["events"].append(cancel_result["message"])
    return format_status(task), task


def run_remote_task(
    api_base_url: str,
    api_key: str,
    start_url: str,
    query: str,
    max_steps: int,
    poll_interval: float,
) -> Generator[Tuple[str, Optional[Image.Image], str, Dict[str, Any]], None, None]:
    if not query.strip():
        raise gr.Error("Task description cannot be empty")

    try:
        created = submit_task(api_base_url, api_key, start_url, query, max_steps)
    except Exception as exc:
        raise gr.Error(f"Failed to submit task: {exc}") from exc

    task_id = created["task_id"]
    intro = f"### Task Submitted\n- Task ID: `{task_id}`\n- Status URL: {created.get('status_url', '-')}"
    yield intro, None, "", {"task_id": task_id, "status": "queued"}, ""

    last_image_url = None
    last_image = None

    while True:
        try:
            task = get_task_status(api_base_url, api_key, task_id)
        except Exception as exc:
            raise gr.Error(f"Failed to query task status: {exc}") from exc

        image_url = task.get("display_image_url") or task.get("latest_annotated_screenshot_url") or task.get("latest_screenshot_url")
        if image_url and image_url != last_image_url:
            try:
                last_image = fetch_image(image_url, api_key)
                last_image_url = image_url
            except Exception:
                pass

        final_answer = task.get("final_answer", "")
        yield format_status(task), last_image, final_answer, task, format_raw_response(task)

        if task.get("status") in {"done", "error", "cancelled"}:
            break

        time.sleep(max(0.5, float(poll_interval)))


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="OpAgent Remote Demo") as demo:
        gr.Markdown("# OpAgent Remote Demo\nModelScope frontend for remote execution on ECS.")

        # with gr.Row():
        #     api_base_url = gr.Textbox(label="ECS API Base URL", value=DEFAULT_API_BASE_URL)
        api_base_url = gr.State(DEFAULT_API_BASE_URL)
        api_key = gr.State(DEFAULT_API_KEY)

        with gr.Row():
            start_url = gr.Textbox(label="Start URL", value=DEFAULT_START_URL, placeholder="https://www.bilibili.com")
            max_steps = gr.Slider(label="Max Steps", minimum=5, maximum=100, value=30, step=1)
            poll_interval = gr.Slider(label="Polling Interval (seconds)", minimum=1, maximum=10, value=DEFAULT_POLL_INTERVAL, step=1)

        query = gr.Textbox(label="Task Description", value=DEFAULT_QUERY, lines=4, placeholder="Describe what you want the browser agent to do")

        with gr.Row():
            submit_button = gr.Button("Submit Task", variant="primary")
            cancel_button = gr.Button("Stop Task", variant="stop")

        with gr.Row():
            status_markdown = gr.Markdown(label="Status")
            latest_image = gr.Image(label="Execution Screenshot", type="pil")

        final_answer = gr.Textbox(label="Final Answer", lines=6)
        raw_status = gr.JSON(label="Raw Status")
        model_output = gr.Textbox(label="Raw Model Output", lines=14)

        submit_button.click(
            fn=run_remote_task,
            inputs=[api_base_url, api_key, start_url, query, max_steps, poll_interval],
            outputs=[status_markdown, latest_image, final_answer, raw_status, model_output],
        )

        cancel_button.click(
            fn=cancel_remote_task,
            inputs=[api_base_url, api_key, raw_status],
            outputs=[status_markdown, raw_status],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
