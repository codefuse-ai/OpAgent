#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Browser session and trajectory recorder for OpAgent.

Supports two browser modes driven by the chosen model backend:

    headless=True   (vllm backend, full 32-B)
        Runs on GPU servers without a display.  The agent can only interact
        with the page through screenshots; trajectory screenshots are the
        only way to observe what happened.

    headless=False  (llama_cpp backend, INT4 quantized)
        Runs on a local machine with a 24 GB GPU.  A visible browser window
        opens so you can watch the agent act in real time.  Trajectory
        screenshots are still saved for later review.

All screenshot-related helpers (annotation, recording) live here so the
main agent loop stays clean.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError, async_playwright


DEFAULT_URL = "https://www.google.com"
MAX_STEPS = 50
VIEWPORT = {"width": 1280, "height": 800}
SCREENSHOT_TIMEOUT_MS = max(1_000, int(os.getenv("OPAGENT_SCREENSHOT_TIMEOUT_MS", "8_000").replace("_", "")))


# ---------------------------------------------------------------------------
# Trajectory recorder
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """Records screenshots, annotated screenshots, and step metadata."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.screenshots_dir = output_dir / "screenshots"
        self.annotated_dir = output_dir / "annotated"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory: Dict[str, Any] = {
            "query": "",
            "start_url": "",
            "start_time": "",
            "end_time": "",
            "total_steps": 0,
            "success": False,
            "final_answer": "",
            "steps": [],
        }

    def set_task_info(self, query: str, start_url: str) -> None:
        self.trajectory["query"] = query
        self.trajectory["start_url"] = start_url
        self.trajectory["start_time"] = datetime.now().isoformat()

    def save_screenshot(self, step: int, screenshot_bytes: bytes) -> str:
        filepath = self.screenshots_dir / f"step_{step:03d}.png"
        filepath.write_bytes(screenshot_bytes)
        return str(filepath)

    def save_annotated_screenshot(
        self,
        step: int,
        screenshot_bytes: bytes,
        action_type: str,
        params: Dict[str, Any],
    ) -> str:
        """Draw a crosshair on the click/type target and a step label at the top."""
        try:
            img = Image.open(BytesIO(screenshot_bytes)).convert("RGBA")
            x = params.get("x") or params.get("coordinate_x")
            y = params.get("y") or params.get("coordinate_y")

            # Semi-transparent header bar
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([0, 0, img.width, 40], fill=(0, 0, 0, 128))
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

            # Crosshair at action coordinates
            if x is not None and y is not None:
                xi, yi = int(x), int(y)
                draw.ellipse([xi - 15, yi - 15, xi + 15, yi + 15], outline=(255, 0, 0, 255), width=3)
                draw.line([(xi - 25, yi), (xi + 25, yi)], fill=(255, 0, 0, 255), width=3)
                draw.line([(xi, yi - 25), (xi, yi + 25)], fill=(255, 0, 0, 255), width=3)
                font_sm = _load_font(16, bold=False)
                draw.text((xi + 20, yi - 10), f"({xi}, {yi})", fill=(255, 0, 0, 255), font=font_sm)

            font_lg = _load_font(18, bold=True)
            draw.text((10, 10), f"Step {step}: {action_type}", fill=(255, 255, 255, 255), font=font_lg)

            filepath = self.annotated_dir / f"step_{step:03d}_annotated.png"
            img.save(filepath, "PNG")
            return str(filepath)
        except Exception as exc:
            logger.warning(f"Failed to create annotated screenshot: {exc}")
            return ""

    def record_step(
        self,
        step: int,
        screenshot_path: str,
        annotated_path: str,
        think_content: str,
        action_type: str,
        params: Dict[str, Any],
        url: str,
        error: str = "",
    ) -> None:
        self.trajectory["steps"].append(
            {
                "step": step,
                "url": url,
                "screenshot": screenshot_path,
                "annotated_screenshot": annotated_path,
                "think": think_content,
                "action_type": action_type,
                "params": params,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.trajectory["total_steps"] = step

    def finish(self, success: bool, final_answer: str = "") -> Path:
        self.trajectory["end_time"] = datetime.now().isoformat()
        self.trajectory["success"] = success
        self.trajectory["final_answer"] = final_answer
        trajectory_path = self.output_dir / "trajectory.json"
        trajectory_path.write_text(
            json.dumps(self.trajectory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return trajectory_path


# ---------------------------------------------------------------------------
# Browser session
# ---------------------------------------------------------------------------

class BrowserSession:
    """A single headless Chromium session backed by Playwright."""

    def __init__(self, start_url: str = DEFAULT_URL, headless: bool = True) -> None:
        self.start_url = start_url
        self.headless = headless
        self._playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self.context = await self.browser.new_context(
            viewport=VIEWPORT,
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self.page = await self.context.new_page()

    async def navigate(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(1)

    async def sync_active_page(self) -> Page:
        """Refresh ``self.page`` when a user action opened or focused another tab."""
        if not self.context:
            return self.page

        pages = [p for p in self.context.pages if not p.is_closed()]
        if not pages:
            return self.page

        active_page = None
        for candidate in reversed(pages):
            try:
                if await candidate.evaluate("document.hasFocus()"):
                    active_page = candidate
                    break
            except Exception:
                continue

        if active_page is None:
            active_page = pages[-1]

        if self.page is None or active_page != self.page:
            self.page = active_page
            try:
                await self.page.bring_to_front()
            except Exception:
                pass
            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:
                pass
            await asyncio.sleep(0.5)

        return self.page

    async def take_screenshot(self) -> bytes:
        await self.sync_active_page()
        try:
            return await self.page.screenshot(
                type="png",
                timeout=SCREENSHOT_TIMEOUT_MS,
                animations="disabled",
            )
        except PlaywrightTimeoutError as exc:
            logger.warning(
                f"Playwright screenshot timed out after {SCREENSHOT_TIMEOUT_MS}ms; "
                f"falling back to CDP capture: {exc}"
            )
            return await self._capture_screenshot_via_cdp()

    async def _capture_screenshot_via_cdp(self) -> bytes:
        if not self.context or not self.page:
            raise RuntimeError("Browser page is not ready for screenshot capture")

        cdp_session = await self.context.new_cdp_session(self.page)
        try:
            result = await cdp_session.send("Page.captureScreenshot", {"format": "png"})
            return base64.b64decode(result["data"])
        finally:
            try:
                await cdp_session.detach()
            except Exception:
                pass

    async def stop(self) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

def _load_font(size: int, *, bold: bool) -> ImageFont.ImageFont:
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()
