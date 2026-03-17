#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OAgent Browser - Web Agent with Playwright browser.

Supports two model backends selected via ``--backend``:

    vllm        Full-precision codefuse-ai/OpAgent (Qwen3-32B) via vLLM.
                Runs on a GPU server (no display required).
                Browser mode: HEADLESS — agent sees only screenshots.
                Trajectory screenshots are saved for every step.

    llama_cpp   INT4-quantized model via a running llama-server HTTP API.
                Runs locally on a 24 GB GPU machine.
                Browser mode: HEADED (visible window) by default.
                Pass --headless to force headless mode.
                Trajectory screenshots are still saved.

Usage examples:
    # Full-precision backend (headless, GPU server)
    python main.py --backend vllm --max-steps 30

    # INT4 quantized backend (headed window, local machine)
    python main.py --backend llama_cpp --max-steps 30

    # INT4 backend forced headless
    python main.py --backend llama_cpp --headless --max-steps 30

    # Non-interactive mode
    python main.py --backend llama_cpp --url https://www.google.com --task "Search for python tutorials"
"""

import asyncio
import base64
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

# Shared utilities
from action_executor import execute_browser_action, is_terminal_action
from browser_runtime import BrowserSession, TrajectoryRecorder, DEFAULT_URL, MAX_STEPS

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")



# =============================================================================
# Backend loader
# =============================================================================

def load_backend(backend: str):
    """Import and return (call_model, ModelInput, load_model_fn, set_config_fn).

    ``load_model_fn`` pre-loads vLLM weights or verifies llama-server health.
    """
    if backend == "vllm":
        from model_interface_vllm import call_model, ModelInput, load_model, set_model_config
        return call_model, ModelInput, load_model, set_model_config
    elif backend == "llama_cpp":
        from model_interface_llama_cpp import call_model, ModelInput, ensure_llama_server_available

        def _noop_load():
            ensure_llama_server_available()
            logger.info("✅ llama-server reachable")

        def _noop_set(**_kwargs):
            pass

        return call_model, ModelInput, _noop_load, _noop_set
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'vllm' or 'llama_cpp'.")


# =============================================================================
# Agent
# =============================================================================

class OAgentBrowser:
    """Web Agent that drives a Playwright browser using a chosen model backend."""

    def __init__(
        self,
        call_model_fn,
        ModelInputClass,
        start_url: str = DEFAULT_URL,
        output_dir: Optional[str] = None,
        max_steps: int = MAX_STEPS,
        headless: bool = True,
    ) -> None:
        self.call_model = call_model_fn
        self.ModelInput = ModelInputClass
        self.start_url = start_url
        self.max_steps = max_steps

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")

        self._session = BrowserSession(start_url=start_url, headless=headless)
        self._action_history: List[Dict[str, Any]] = []

    async def start(self) -> None:
        await self._session.start()
        logger.info(f"🚀 Browser started  (headless={self._session.headless})")

    async def stop(self) -> None:
        await self._session.stop()
        logger.info("👋 Browser stopped")

    async def navigate(self, url: str) -> None:
        await self._session.navigate(url)

    @property
    def page(self):
        return self._session.page

    async def execute_task(self, query: str, task_url: Optional[str] = None) -> Path:
        """Run the agent loop for one task.

        Args:
            query:    Natural-language task description.
            task_url: Optional URL to navigate to before starting.

        Returns:
            Path to the saved trajectory JSON.
        """
        logger.info(f"\n{'='*60}\n📋 Task: {query}")
        if task_url:
            logger.info(f"🌐 Task URL: {task_url}")
        logger.info(f"{'='*60}\n")

        task_output_dir = self.output_dir / datetime.now().strftime("%H%M%S")
        recorder = TrajectoryRecorder(task_output_dir)
        recorder.set_task_info(query, task_url or self._session.page.url)

        self._action_history = []

        if task_url:
            await self._session.navigate(task_url)

        final_answer = ""
        success = False

        try:
            for step in range(1, self.max_steps + 1):
                logger.info(f"\n{'─'*40}\n📍 Step {step}/{self.max_steps}\n{'─'*40}")

                screenshot_bytes = await self._session.take_screenshot()
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                screenshot_path = recorder.save_screenshot(step, screenshot_bytes)
                logger.info("📸 Screenshot taken")

                logger.info("🤖 Calling model …")
                model_input = self.ModelInput(
                    screenshot_base64=screenshot_base64,
                    query=query,
                    history=self._action_history,
                    current_url=self._session.page.url,
                )
                model_output = self.call_model(model_input)

                if model_output.think_content:
                    logger.info(f"💭 Think: {model_output.think_content[:200]}")

                if not model_output.tool_call:
                    logger.warning("❌ No valid action from model")
                    recorder.record_step(
                        step=step,
                        screenshot_path=screenshot_path,
                        annotated_path="",
                        think_content=model_output.think_content or "",
                        action_type="error",
                        params={"error": "No valid action"},
                        url=self._session.page.url,
                        error="No valid action from model",
                    )
                    continue

                tool_call = model_output.tool_call
                action_type = tool_call.get("action_type", "")
                logger.info(f"🔧 Action: {action_type}")
                logger.info(f"📋 Params: {json.dumps(tool_call, ensure_ascii=False)}")

                annotated_path = recorder.save_annotated_screenshot(step, screenshot_bytes, action_type, tool_call)

                if is_terminal_action(action_type):
                    final_answer = tool_call.get("content", tool_call.get("answer", "Task completed"))
                    logger.info(f"✅ Task completed!  Answer: {final_answer}")
                    recorder.record_step(
                        step=step,
                        screenshot_path=screenshot_path,
                        annotated_path=annotated_path,
                        think_content=model_output.think_content or "",
                        action_type=action_type,
                        params=tool_call,
                        url=self._session.page.url,
                    )
                    success = True
                    break

                error_msg = await execute_browser_action(self._session.page, action_type, tool_call)
                if error_msg:
                    logger.warning(f"⚠️  Action error: {error_msg}")

                recorder.record_step(
                    step=step,
                    screenshot_path=screenshot_path,
                    annotated_path=annotated_path,
                    think_content=model_output.think_content or "",
                    action_type=action_type,
                    params=tool_call,
                    url=self._session.page.url,
                    error=error_msg,
                )

                self._action_history.append({
                    "step": step,
                    "action_type": action_type,
                    "params": tool_call,
                    "error": error_msg,
                    "url": self._session.page.url,
                })

                await asyncio.sleep(0.5)

            else:
                logger.warning(f"⚠️  Reached max steps ({self.max_steps})")

        except KeyboardInterrupt:
            logger.info("\n⏹️  Task interrupted by user")
        except Exception as e:
            logger.error(f"❌ Task execution error: {e}")
            import traceback
            traceback.print_exc()

        trajectory_path = recorder.finish(success, final_answer)
        return trajectory_path


# =============================================================================
# CLI entry point
# =============================================================================

def print_banner(backend: str) -> None:
    model_label = (
        "codefuse-ai/OpAgent (Qwen3-32B, full precision via vLLM)"
        if backend == "vllm"
        else "OpAgent-32B-Q4 (INT4 quantized via llama-server)"
    )
    print(f"""
╔═══════════════════════════════════════════════════════╗
║           OAgent Browser  –  Web Agent               ║
║  Backend : {backend:<12}                              ║
║  Model   : {model_label[:43]:<43} ║
╚═══════════════════════════════════════════════════════╝
""")


async def main() -> None:
    parser = argparse.ArgumentParser(description="OAgent Browser - Headless Web Agent")
    parser.add_argument(
        "--backend", "-b",
        choices=["vllm", "llama_cpp"],
        default="vllm",
        help="Model backend: 'vllm' (full 32B, GPU-heavy) or 'llama_cpp' (INT4, lightweight). Default: vllm",
    )
    parser.add_argument("--output", "-o", default=None, help="Output directory for trajectory files")
    parser.add_argument("--max-steps", "-m", type=int, default=MAX_STEPS, help=f"Max steps per task (default: {MAX_STEPS})")
    parser.add_argument("--url", default=None, help="Task start URL (non-interactive mode)")
    parser.add_argument("--task", default=None, help="Task description (non-interactive mode)")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=None,
        help="Force headless browser mode (default: headless for vllm, headed for llama_cpp)",
    )
    parser.add_argument("--model-path", default=None, help="[vllm] Local model path")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=None, help="[vllm] Tensor parallelism GPU count")
    parser.add_argument("--max-model-len", type=int, default=None, help="[vllm] Max model sequence length (default: 32768)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="[vllm] GPU memory utilization (default: 0.9)")
    args = parser.parse_args()

    print_banner(args.backend)

    call_model_fn, ModelInputClass, load_model_fn, set_config_fn = load_backend(args.backend)

    if args.backend == "vllm" and any([args.model_path, args.tensor_parallel_size, args.max_model_len, args.gpu_memory_utilization]):
        set_config_fn(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    print("\n" + "=" * 60)
    print(f"🔄 Initialising backend: {args.backend} …")
    print("=" * 60)
    try:
        load_model_fn()
    except Exception as e:
        print(f"❌ Backend initialisation failed: {e}")
        if args.backend == "vllm":
            print("Tips: --max-model-len 16384  |  --gpu-memory-utilization 0.95  |  --tensor-parallel-size 2")
        else:
            print("Make sure llama-server is running and OPAGENT_LLAMA_SERVER_URL is set correctly.")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("🌐 Starting browser …")
    print("=" * 60)

    # Determine headless mode:
    #   vllm      → always headless (GPU server, no display)
    #   llama_cpp → headed by default (local 24 GB machine, visible window)
    #              override with --headless flag
    if args.headless is not None:
        use_headless = args.headless  # explicit flag wins
    else:
        use_headless = (args.backend == "vllm")

    mode_desc = "headless" if use_headless else "headed (visible window)"
    print(f"🖥️  Browser mode: {mode_desc}")

    agent = OAgentBrowser(
        call_model_fn=call_model_fn,
        ModelInputClass=ModelInputClass,
        start_url=DEFAULT_URL,
        output_dir=args.output,
        max_steps=args.max_steps,
        headless=use_headless,
    )

    try:
        await agent.start()
        await agent.navigate(DEFAULT_URL)

        print(f"\n📁 Output directory: {agent.output_dir}")
        print(f"🔢 Max steps: {args.max_steps}")

        if args.url or args.task:
            query = args.task or ""
            if not query:
                print("⚠️  --task is required in non-interactive mode")
            else:
                trajectory = await agent.execute_task(query, args.url)
                print(f"\n📁 Trajectory saved: {trajectory}")
            return

        print("\n" + "=" * 60)
        while True:
            print()
            try:
                url_input = input("🌐 Enter task URL (Enter = current page, 'quit' to exit): ").strip()
                if url_input.lower() in {"quit", "exit", "q"}:
                    break

                task_url = None
                if url_input:
                    task_url = url_input if url_input.startswith(("http://", "https://")) else f"https://{url_input}"
                    try:
                        print(f"   → Navigating to: {task_url}")
                        await agent.navigate(task_url)
                        print("   ✅ URL loaded")
                    except Exception as e:
                        print(f"   ⚠️  Failed to load URL: {e}")
                        task_url = None
                else:
                    print(f"   → Using current page: {agent.page.url}")

                query = input("🎯 Enter task description: ").strip()
                if not query or query.lower() in {"quit", "exit", "q"}:
                    break

                trajectory = await agent.execute_task(query, None)
                print(f"\n📁 Trajectory: {trajectory}")

            except EOFError:
                break

    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
