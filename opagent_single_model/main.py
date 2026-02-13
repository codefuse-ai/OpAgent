#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OAgent Browser - Headless Browser Web Agent

A Web Agent tool based on Playwright headless browser, suitable for server-side execution.

Features:
1. Interactive task input via terminal
2. Local OpAgent model invocation (codefuse-ai/OpAgent, based on Qwen3-32B)
3. Parse model output: <think>...</think><tool_call>...</tool_call>
4. Headless browser action execution
5. Save complete trajectory: screenshots, think, action, annotated screenshots

Usage:
    python main.py [--output OUTPUT_DIR] [--max-steps MAX_STEPS]

Example:
    python main.py --max-steps 30
"""

import os
import sys
import json
import asyncio
import base64
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Import local modules
from action_executor import execute_browser_action, is_terminal_action
from model_interface import call_model, ModelInput, ModelOutput, set_model_config


# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


# Default configuration
DEFAULT_URL = "https://www.google.com"  # Default start page
MAX_STEPS = 50
VIEWPORT = {'width': 1280, 'height': 800}


class TrajectoryRecorder:
    """Trajectory Recorder - Records screenshot, thinking, and action for each step"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.screenshots_dir = output_dir / "screenshots"
        self.annotated_dir = output_dir / "annotated"
        
        # Create directories
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # Trajectory data
        self.trajectory = {
            "query": "",
            "start_url": "",
            "start_time": "",
            "end_time": "",
            "total_steps": 0,
            "success": False,
            "final_answer": "",
            "steps": []
        }
    
    def set_task_info(self, query: str, start_url: str):
        """Set task information"""
        self.trajectory["query"] = query
        self.trajectory["start_url"] = start_url
        self.trajectory["start_time"] = datetime.now().isoformat()
    
    def save_screenshot(self, step: int, screenshot_bytes: bytes) -> str:
        """Save original screenshot"""
        filename = f"step_{step:03d}.png"
        filepath = self.screenshots_dir / filename
        with open(filepath, 'wb') as f:
            f.write(screenshot_bytes)
        return str(filepath)
    
    def save_annotated_screenshot(self, step: int, screenshot_bytes: bytes, 
                                   action_type: str, params: Dict) -> str:
        """Save annotated screenshot"""
        try:
            # Load image and convert to RGBA for transparency support
            img = Image.open(BytesIO(screenshot_bytes)).convert("RGBA")
            
            # Get coordinates
            x = params.get('x') or params.get('coordinate_x')
            y = params.get('y') or params.get('coordinate_y')
            
            # Create a transparent overlay for semi-transparent background
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Draw semi-transparent background bar (alpha=128, ~50% transparent)
            overlay_draw.rectangle([0, 0, img.width, 40], fill=(0, 0, 0, 128))
            
            # Merge overlay
            img = Image.alpha_composite(img, overlay)
            
            # Draw on the merged image
            draw = ImageDraw.Draw(img)
            
            if x is not None and y is not None:
                x, y = int(x), int(y)
                
                # Draw crosshair and circle to mark click position
                radius = 15
                cross_size = 25
                color = (255, 0, 0, 255)  # Red
                width = 3
                
                # Draw circle
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    outline=color, width=width
                )
                
                # Draw crosshair
                draw.line([(x - cross_size, y), (x + cross_size, y)], fill=color, width=width)
                draw.line([(x, y - cross_size), (x, y + cross_size)], fill=color, width=width)
                
                # Add coordinate text
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                    except:
                        font = ImageFont.load_default()
                
                coord_text = f"({x}, {y})"
                draw.text((x + 20, y - 10), coord_text, fill=color, font=font)
            
            # Add action info at top of image (show full parameters)
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except:
                try:
                    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
                except:
                    font_large = ImageFont.load_default()
            
            # Build complete action description
            action_text = f"Step {step}: {action_type}"
            param_parts = []
            
            # Add coordinate parameters
            if x is not None and y is not None:
                param_parts.append(f"coords=[{x},{y}]")
            
            # Add other important parameters
            if 'text' in params:
                text_val = params['text']
                if len(text_val) > 25:
                    text_val = text_val[:25] + "..."
                param_parts.append(f"content='{text_val}'")
            elif 'content' in params:
                content_val = params['content']
                if len(content_val) > 25:
                    content_val = content_val[:25] + "..."
                param_parts.append(f"content='{content_val}'")
            
            if 'direction' in params:
                param_parts.append(f"direction={params['direction']}")
            if 'distance' in params:
                param_parts.append(f"distance={params['distance']}")
            if 'key' in params:
                param_parts.append(f"key='{params['key']}'")
            if 'url' in params:
                url_val = params['url']
                if len(url_val) > 30:
                    url_val = url_val[:30] + "..."
                param_parts.append(f"url='{url_val}'")
            if 'seconds' in params:
                param_parts.append(f"seconds={params['seconds']}")
            if 'press_enter' in params:
                param_parts.append(f"press_enter={1 if params['press_enter'] else 0}")
            
            if param_parts:
                action_text += f"({', '.join(param_parts)})"
            
            draw.text((10, 10), action_text, fill=(255, 255, 255, 255), font=font_large)
            
            # Save as PNG (supports transparency)
            filename = f"step_{step:03d}_annotated.png"
            filepath = self.annotated_dir / filename
            img.save(filepath, "PNG")
            
            return str(filepath)
            
        except Exception as e:
            logger.warning(f"Failed to create annotated screenshot: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def record_step(self, step: int, screenshot_path: str, annotated_path: str,
                    think_content: str, action_type: str, params: Dict,
                    url: str, error: str = ""):
        """Record one step"""
        step_data = {
            "step": step,
            "url": url,
            "screenshot": screenshot_path,
            "annotated_screenshot": annotated_path,
            "think": think_content,
            "action_type": action_type,
            "params": params,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.trajectory["steps"].append(step_data)
        self.trajectory["total_steps"] = step
    
    def finish(self, success: bool, final_answer: str = ""):
        """Finish recording"""
        self.trajectory["end_time"] = datetime.now().isoformat()
        self.trajectory["success"] = success
        self.trajectory["final_answer"] = final_answer
        
        # Save trajectory JSON
        trajectory_path = self.output_dir / "trajectory.json"
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(self.trajectory, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Trajectory saved to: {trajectory_path}")
        
        return trajectory_path


class OAgentBrowser:
    """OAgent Headless Browser Controller"""
    
    def __init__(self, start_url: str = DEFAULT_URL, output_dir: str = None, max_steps: int = MAX_STEPS):
        self.start_url = start_url
        self.max_steps = max_steps
        
        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Playwright
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Task state
        self.action_history: List[Dict[str, Any]] = []
        
    async def start(self):
        """Start browser"""
        logger.info("=" * 60)
        logger.info("🚀 OAgent Browser (Headless Mode)")
        logger.info("=" * 60)
        
        self.playwright = await async_playwright().start()
        
        # Launch headless browser
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )
        
        # Create browser context
        self.context = await self.browser.new_context(
            viewport=VIEWPORT,
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # Create page
        self.page = await self.context.new_page()
        
        logger.info(f"🌐 Headless browser started")
        logger.info(f"📐 Viewport: {VIEWPORT['width']}x{VIEWPORT['height']}")
    
    async def navigate(self, url: str):
        """Navigate to specified URL"""
        logger.info(f"🔗 Navigating to: {url}")
        await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
        await asyncio.sleep(1)  # Wait for page to stabilize
    
    async def take_screenshot(self) -> bytes:
        """Take screenshot"""
        return await self.page.screenshot(type='png')
    
    async def execute_task(self, query: str, task_url: str = None) -> Path:
        """Execute task
        
        Args:
            query: Task description
            task_url: Optional task start URL, will navigate to this URL if specified
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 Task: {query}")
        if task_url:
            logger.info(f"🌐 Task URL: {task_url}")
        logger.info(f"{'='*60}\n")
        
        # Create new output directory for each task
        task_output_dir = self.output_dir / datetime.now().strftime("%H%M%S")
        
        # Create trajectory recorder
        recorder = TrajectoryRecorder(task_output_dir)
        recorder.set_task_info(query, task_url or self.page.url)
        
        # Reset history
        self.action_history = []
        
        # Navigate to task URL if specified
        if task_url:
            await self.navigate(task_url)
        
        final_answer = ""
        success = False
        
        try:
            for step in range(1, self.max_steps + 1):
                logger.info(f"\n{'─'*40}")
                logger.info(f"📍 Step {step}/{self.max_steps}")
                logger.info(f"{'─'*40}")
                
                # 1. Take screenshot
                screenshot_bytes = await self.take_screenshot()
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                
                # Save original screenshot
                screenshot_path = recorder.save_screenshot(step, screenshot_bytes)
                logger.info(f"📸 Screenshot saved")
                
                # 2. Call model
                logger.info(f"🤖 Calling model...")
                model_input = ModelInput(
                    screenshot_base64=screenshot_base64,
                    query=query,
                    history=self.action_history,
                    current_url=self.page.url
                )
                
                model_output = call_model(model_input)
                
                # 3. Display thinking process
                if model_output.think_content:
                    logger.info(f"💭 Think: {model_output.think_content[:200]}...")
                
                # 4. Check if there's a valid action
                if not model_output.tool_call:
                    logger.warning("❌ No valid action from model")
                    recorder.record_step(
                        step=step,
                        screenshot_path=screenshot_path,
                        annotated_path="",
                        think_content=model_output.think_content or "",
                        action_type="error",
                        params={"error": "No valid action"},
                        url=self.page.url,
                        error="No valid action from model"
                    )
                    continue
                
                tool_call = model_output.tool_call
                action_type = tool_call.get('action_type', '')
                
                logger.info(f"🔧 Action: {action_type}")
                logger.info(f"📋 Params: {json.dumps(tool_call, ensure_ascii=False)}")
                
                # 5. Save annotated screenshot
                annotated_path = recorder.save_annotated_screenshot(
                    step, screenshot_bytes, action_type, tool_call
                )
                
                # 6. Check if it's a terminal action
                if is_terminal_action(action_type):
                    final_answer = tool_call.get('content', tool_call.get('answer', 'Task completed'))
                    logger.info(f"✅ Task completed!")
                    logger.info(f"📝 Answer: {final_answer}")
                    
                    recorder.record_step(
                        step=step,
                        screenshot_path=screenshot_path,
                        annotated_path=annotated_path,
                        think_content=model_output.think_content or "",
                        action_type=action_type,
                        params=tool_call,
                        url=self.page.url
                    )
                    success = True
                    break
                
                # 7. Execute browser action
                error_msg = await execute_browser_action(self.page, action_type, tool_call)
                
                if error_msg:
                    logger.warning(f"⚠️ Action error: {error_msg}")
                
                # 8. Record step
                recorder.record_step(
                    step=step,
                    screenshot_path=screenshot_path,
                    annotated_path=annotated_path,
                    think_content=model_output.think_content or "",
                    action_type=action_type,
                    params=tool_call,
                    url=self.page.url,
                    error=error_msg
                )
                
                # 9. Record history
                self.action_history.append({
                    'step': step,
                    'action_type': action_type,
                    'params': tool_call,
                    'error': error_msg,
                    'url': self.page.url
                })
                
                # Wait for page to stabilize
                await asyncio.sleep(0.5)
            
            else:
                # Reached max steps
                logger.warning(f"⚠️ Reached max steps ({self.max_steps})")
        
        except KeyboardInterrupt:
            logger.info("\n⏹️ Task interrupted by user")
        except Exception as e:
            logger.error(f"❌ Task execution error: {e}")
            import traceback
            traceback.print_exc()
        
        # Finish recording
        trajectory_path = recorder.finish(success, final_answer)
        
        return trajectory_path
    
    async def stop(self):
        """Stop browser"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("👋 Browser stopped")


def print_banner():
    """Print banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║      ██████╗ ██████╗  █████╗  ██████╗ ███████╗███╗   ██╗████████╗ ║
║     ██╔═══██╗██╔══██╗██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝ ║
║     ██║   ██║██████╔╝███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║    ║
║     ██║   ██║██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║    ║
║     ╚██████╔╝██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║    ║
║      ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ║
║                                                                   ║
║            Web Agent with Headless Browser                        ║
║            Model: codefuse-ai/OpAgent (Qwen3-32B)                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main():
    parser = argparse.ArgumentParser(description="OAgent Browser - Headless Web Agent")
    parser.add_argument("--output", "-o", default=None, help="Output directory for trajectory")
    parser.add_argument("--max-steps", "-m", type=int, default=MAX_STEPS, help=f"Max steps (default: {MAX_STEPS})")
    parser.add_argument("--model-path", default=None, help="Path to local model (e.g., codefuse-ai/OpAgent)")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=None, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model sequence length (default: 32768)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU memory utilization (default: 0.9)")
    args = parser.parse_args()
    
    print_banner()
    
    # Update configuration if model parameters are specified
    if args.model_path or args.tensor_parallel_size or args.max_model_len or args.gpu_memory_utilization:
        set_model_config(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    
    # ========== Load model first ==========
    print("\n" + "="*60)
    print("🔄 Step 1: Loading model...")
    print("="*60)
    
    try:
        from model_interface import load_model
        load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Please check your GPU memory and model configuration.")
        print("You can try:")
        print("  --max-model-len 16384  (reduce sequence length)")
        print("  --gpu-memory-utilization 0.95  (increase GPU memory usage)")
        print("  --tensor-parallel-size 2  (use more GPUs)")
        import traceback
        traceback.print_exc()
        return
    
    # ========== Then start browser ==========
    print("\n" + "="*60)
    print("🌐 Step 2: Starting browser...")
    print("="*60)
    
    # Create Agent
    agent = OAgentBrowser(
        start_url=DEFAULT_URL,
        output_dir=args.output,
        max_steps=args.max_steps
    )
    
    try:
        # Start browser
        await agent.start()
        
        # Navigate to default page
        await agent.navigate(DEFAULT_URL)
        
        print(f"\n🌐 Default page: {DEFAULT_URL}")
        print(f"📁 Output directory: {agent.output_dir}")
        print(f"🔢 Max steps: {args.max_steps}")
        print("\n" + "="*60)
        
        # Interaction loop
        while True:
            print("\n")
            try:
                # Ask for URL
                url_input = input("🌐 Enter task URL (press Enter to use current page, 'quit' to exit): ").strip()
                
                if url_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                # If URL is provided, prepare to navigate
                task_url = None
                if url_input:
                    task_url = url_input
                    if not task_url.startswith(('http://', 'https://')):
                        task_url = 'https://' + task_url
                    
                    # Validate URL by attempting to navigate
                    try:
                        print(f"   → Navigating to: {task_url}")
                        await agent.navigate(task_url)
                        print(f"   ✅ URL loaded successfully")
                    except Exception as e:
                        print(f"   ⚠️  Failed to load URL: {e}")
                        print(f"   → Using default page: {DEFAULT_URL}")
                        try:
                            await agent.navigate(DEFAULT_URL)
                            task_url = None  # Will use current page (default)
                        except:
                            pass
                else:
                    print(f"   → Using current page: {agent.page.url}")
                
                # Ask for task
                query = input("🎯 Enter task description: ").strip()
                
                if not query:
                    print("⚠️  Task cannot be empty")
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Execute task (URL already loaded, so don't pass task_url to avoid double navigation)
                trajectory_path = await agent.execute_task(query, None)
                
                print(f"\n{'='*60}")
                print(f"📁 Trajectory saved: {trajectory_path}")
                print(f"{'='*60}")
                
            except EOFError:
                break
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
    
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
