"""
Browser Action Executor

Responsible for parsing tool_call and executing browser operations, consistent with prompt.py.

Supported action types (consistent with prompt.py):
- click: Click
- type: Input text (coords, content, press_enter_after)
- hover: Hover
- press: Key press
- scroll: Vertical scroll (up/down)
- hscroll: Horizontal scroll (left/right)
- new_tab: Open new tab
- tab_focus: Switch tab
- go_back: Go back
- go_forward: Go forward
- move_to: Move cursor
- double_click: Double click
- goto: Navigate to URL
- wait: Wait
- answer: Task complete (return answer)
"""

import asyncio
import json
import re
from typing import Dict, Any, Optional, Tuple
from playwright.async_api import Page
from loguru import logger


def get_coords_from_params(params: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Extract coordinates from parameters"""
    # Support multiple coordinate formats
    if "x" in params and "y" in params:
        return params["x"], params["y"]
    if "coordinate_x" in params and "coordinate_y" in params:
        return params["coordinate_x"], params["coordinate_y"]
    if "coords" in params:
        coords = params["coords"]
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return coords[0], coords[1]
    return None, None


def parse_tool_call(model_response: str) -> Optional[Dict]:
    """
    Parse model-returned tool_call
    
    Supported formats:
    <tool_call>{"action_type": "click", "x": 100, "y": 200}</tool_call>
    or JSON format
    """
    try:
        # Try parsing <tool_call> format
        if "<tool_call>" in model_response and "</tool_call>" in model_response:
            tool_call_str = model_response.split("<tool_call>")[1].split("</tool_call>")[0]
            tool_call = json.loads(tool_call_str.strip())
            return tool_call
        
        # Try parsing pure JSON format
        json_match = re.search(r'\{.*\}', model_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        
        return None
    except Exception as e:
        logger.warning(f"Failed to parse tool_call: {e}")
        return None


def parse_think_content(model_response: str) -> Optional[str]:
    """
    Parse model-returned <think> content
    """
    try:
        if "<think>" in model_response and "</think>" in model_response:
            think_content = model_response.split("<think>")[1].split("</think>")[0]
            return think_content.strip()
        return None
    except Exception:
        return None


async def execute_browser_action(page: Page, action_type: str, params: Dict[str, Any]) -> str:
    """
    Execute browser action
    
    Args:
        page: Playwright Page object
        action_type: Action type
        params: Action parameters
        
    Returns:
        error_msg: Error message, empty string if successful
    """
    error_msg = ""
    x, y = get_coords_from_params(params)
    
    try:
        if action_type == "click":
            if x is not None and y is not None:
                logger.info(f"Click at: ({x}, {y})")
                await page.mouse.click(x, y)
            elif "selector" in params:
                logger.info(f"Click selector: {params['selector']}")
                await page.click(params["selector"], timeout=5000)
            else:
                logger.warning(f"Click action missing coords: {params}")
                error_msg = "Click action missing coords"
                
        elif action_type == "double_click":
            if x is not None and y is not None:
                logger.info(f"Double click at: ({x}, {y})")
                await page.mouse.dblclick(x, y)
            else:
                error_msg = "Double click action missing coords"
                
        elif action_type in ["input", "type"]:
            content = params.get("text", params.get("content", ""))
            press_enter = params.get("press_enter", params.get("press_enter_after", False))
            clear_before = params.get("clear_before_input", True)
            
            logger.info(f"Type text: '{content}', press_enter={press_enter}")
            
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
            
            logger.info(f"Scroll: direction={direction}, distance={distance}")
            
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
                logger.info(f"Navigate to: {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            else:
                error_msg = "Goto action missing url"
                
        elif action_type == "press":
            key = params.get("key", "Enter")
            logger.info(f"Press key: {key}")
            if x is not None and y is not None:
                await page.mouse.click(x, y)
            await page.keyboard.press(key)
            
        elif action_type in ["hover", "move_to"]:
            if x is not None and y is not None:
                logger.info(f"Hover at: ({x}, {y})")
                await page.mouse.move(x, y)
            else:
                error_msg = "Hover/move_to action missing coords"
        
        elif action_type == "wait":
            # Wait for a period of time
            wait_time = params.get("time", params.get("seconds", 2))
            logger.info(f"Wait: {wait_time} seconds")
            await asyncio.sleep(wait_time)
                
        elif action_type == "hscroll":
            # Horizontal scroll
            direction = params.get("direction", "right")
            distance = params.get("pixel_amount", params.get("distance", 300))
            
            logger.info(f"Horizontal scroll: direction={direction}, distance={distance}")
            
            # Move to position if coordinates provided
            if x is not None and y is not None:
                await page.mouse.move(x, y)
            
            if direction == "right":
                await page.evaluate(f"window.scrollBy({distance}, 0)")
            elif direction == "left":
                await page.evaluate(f"window.scrollBy(-{distance}, 0)")
                
        elif action_type == "go_back":
            logger.info("Go back")
            await page.go_back()
            
        elif action_type == "go_forward":
            logger.info("Go forward")
            await page.go_forward()
        
        elif action_type == "new_tab":
            # Open new tab - requires context support
            logger.info("Open new tab")
            # Note: Can only create new page here, need external context
            # Single page cannot create new tab, return error message
            error_msg = "new_tab action requires browser context (not supported in single-page mode)"
            logger.warning(error_msg)
        
        elif action_type == "tab_focus":
            # Switch tab - requires context support
            tab_index = params.get("tab_index", 0)
            logger.info(f"Switch to tab: {tab_index}")
            # Single-page mode does not support tab switching
            error_msg = "tab_focus action requires browser context (not supported in single-page mode)"
            logger.warning(error_msg)
            
        elif action_type == "select_option":
            # Special handling for select_option, consistent with local_agent_eval_shopping_admin_en.py
            option_text = params.get("option", params.get("value", ""))
            
            if x is not None and y is not None and option_text:
                logger.info(f"Select option: '{option_text}' at ({x}, {y})")
                try:
                    # Step 1: Click coordinates to force open dropdown menu
                    await page.mouse.click(x, y)
                    await asyncio.sleep(0.3)
                    
                    # Step 2: Use JS to find select element and generate selector
                    target_selector = await page.evaluate(f'''() => {{
                        const el = document.elementFromPoint({x}, {y});
                        if (!el) return null;
                        
                        // Search upward until finding <select> or body
                        let target = el;
                        for (let i=0; i<5; i++) {{ // Search up to 5 levels
                            if (target.tagName === 'SELECT') break;
                            if (!target.parentElement || target.parentElement.tagName === 'BODY') break;
                            target = target.parentElement;
                        }}
                        
                        // If <select> not found, use original clicked element
                        if (target.tagName !== 'SELECT') {{
                            target = el; 
                        }}
                        
                        // Generate a unique CSS selector
                        if (target.id) return `#${{target.id}}`;
                        if (target.name) return `[name="${{target.name}}"]`;
                        return null;
                    }}''')
                    
                    if target_selector:
                        # Use Playwright's select_option
                        await page.select_option(target_selector, label=option_text, timeout=3000)
                        logger.info(f"Successfully selected with Playwright select_option: {option_text}")
                    else:
                        raise Exception("Could not generate a selector for the element at the coordinates.")

                except Exception as e:
                    # If Playwright method fails, fallback to JS method
                    logger.warning(f"Playwright select_option failed: {e}. Falling back to JS method.")
                    try:
                        # Handle escaping outside f-string to avoid backslash in braces
                        escaped_option_text = option_text.replace("'", "\\'")
                        js_code = f'''
                        (async (optionText) => {{
                            let el = document.elementFromPoint({x}, {y});
                            if (!el) return false;
                            
                            // If clicked element is not select, search upward
                            if (el.tagName !== 'SELECT') {{
                                const parentSelect = el.closest('select');
                                if (parentSelect) el = parentSelect;
                            }}

                            if (el && el.tagName === 'SELECT') {{
                                for (const option of el.options) {{
                                    // 1. First try exact text match
                                    if (option.text.trim() === optionText.trim()) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                }}
                                for (const option of el.options) {{
                                    // 2. Try contains text match
                                    if (option.text.trim().includes(optionText.trim())) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                }}
                                for (const option of el.options) {{
                                    // 3. Try matching value
                                    if (option.value.trim() === optionText.trim()) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                }}
                            }}
                            return false;
                        }})('{escaped_option_text}')
                        '''
                        success = await page.evaluate(js_code)
                        if not success:
                            error_msg = f"JS fallback for select_option also failed for option: {option_text}"
                            logger.error(error_msg)
                        else:
                            logger.info(f"Successfully selected with JS method: {option_text}")
                            
                    except Exception as e2:
                        error_msg = f"JS fallback for select_option threw an error: {e2}"
                        logger.error(error_msg)
            else:
                error_msg = f"select_option missing coords or option text. Params: {params}"
                logger.warning(error_msg)
                
        elif action_type == "right_click":
            if x is not None and y is not None:
                logger.info(f"Right click at: ({x}, {y})")
                await page.mouse.click(x, y, button="right")
            else:
                error_msg = "Right click action missing coords"

        elif action_type in ["stop", "finish", "done", "answer"]:
            logger.info(f"Task completed: {action_type}")
            # Terminal action, no operation needed
            pass

        else:
            error_msg = f"Unknown action type: {action_type}"
            logger.warning(error_msg)

        # Wait for page to stabilize
        if action_type not in ["stop", "finish", "done", "answer"]:
            await asyncio.sleep(0.5)
            try:
                await page.wait_for_load_state('networkidle', timeout=5000)
            except:
                pass
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Action execution error: {e}")
    
    return error_msg


def is_terminal_action(action_type: str) -> bool:
    """Check if it's a terminal action"""
    return action_type in ["stop", "finish", "done", "answer"]
