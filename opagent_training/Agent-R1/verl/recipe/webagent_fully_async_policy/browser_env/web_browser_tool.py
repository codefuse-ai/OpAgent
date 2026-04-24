"""
Search tool implementation for simulating internet searches
"""

import time
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
from recipe.webagent_fully_async_policy.browser_env.tool_base import Tool
from recipe.webagent_fully_async_policy.browser_env.envs import ScriptBrowserEnv
from recipe.webagent_fully_async_policy.browser_env.async_web_browser_envs import  AsyncScriptBrowserEnv
from recipe.webagent_fully_async_policy.browser_env.actions import *
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures
from recipe.webagent_fully_async_policy.browser_env.utils import smart_resize, calculate_ssim, draw_image_with_coords
from recipe.webagent_fully_async_policy.evaluation_harness.webjudge.src.qwen_vlm_request import compare_two_images_with_shangshu
import logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', "")

global_count = 0
VLM_EXP_DEBUG = os.environ.get('VLM_EXP_DEBUG', '0')
VLM_EXP_DEBUG = str(VLM_EXP_DEBUG)

OBSERVATION_TYPE = os.environ.get('OBSERVATION_TYPE', '')

print(f"!!!!!VLM_EXP_DEBUG: {VLM_EXP_DEBUG}")
MAX_CONCURRENT_WORKERS = int(os.environ.get('MAX_CONCURRENT_WORKERS', 32)) 


class WebBrowserTool(Tool):
    """
    Tool for searching Wikipedia using the wiki_search_api
    """
    
    def __init__(self, name, description, parameters):
        """
        Initialize the web browser tool
        """
        super().__init__(name, description, parameters)
        
    
    def convert_action_to_string_id(self, properties: Dict):
        """
        Convert a JSON-formatted action to a string representation.
        
        :param properties: A dictionary of action properties
        :return: A string representation of the action
        :raises KeyError: If a required property is missing
        """
        name = self.name
        
        try:
            if name == 'click':
                return f"click [{properties['id']}]"
            
            elif name == 'type':
                # 检查必需的参数
                if 'press_enter_after' in properties:
                    # 如果明确指定了press_enter_after
                    return f"type [{properties['id']}] [{properties['content']}] [{properties['press_enter_after']}]"
                else:
                    # 默认情况下按回车
                    return f"type [{properties['id']}] [{properties['content']}]"
            
            elif name == 'hover':
                return f"hover [{properties['id']}]"
            
            elif name == 'press':
                return f"press [{properties['key_combination']}]"
            
            elif name == 'scroll':
                return f"scroll [{properties['direction']}]"
            
            elif name == 'new_tab':
                return "new_tab"
            
            elif name == 'tab_focus':
                return f"tab_focus [{properties['tab_index']}]"
            
            elif name == 'close_tab':
                return "close_tab"
            
            elif name == 'goto':
                return f"goto [{properties['url']}]"
            
            elif name == 'go_back':
                return "go_back"
            
            elif name == 'go_forward':
                return "go_forward"
            
            # 如果无法识别action
            return None
        
        except KeyError as e:
            # 抛出具体缺失的属性异常
            #raise KeyError(f"Missing required property {e} for action {name}")
            print(f"Missing required property {e} for action {name}")
            return None


    def convert_action_to_string_coords(self, properties: Dict, env_i: AsyncScriptBrowserEnv) -> Optional[str]:
        """
        Convert a JSON-formatted action to a string representation, converting
        coordinates and distances from resized dimensions back to original dimensions.
        
        :param properties: A dictionary of action properties with values corresponding to the RESIZED image.
        :param env_i: The browser environment instance providing original viewport size.
        :return: A string representation of the action with values scaled to the ORIGINAL viewport, or None on error.
        :raises KeyError: If a required property is missing (handled internally).
        """
        name = self.name
        
        # 1. Get original and resized dimensions
        original_width = env_i.viewport_size['width']
        original_height = env_i.viewport_size['height']
        
        # 假设 smart_resize 是可用的。您可能需要从您的代码库中导入它。
        # smart_resize 的参数是示例性的，请根据您的实际情况调整。
        resized_height, resized_width = smart_resize(original_height, original_width, 32, 512 * 512, 2048 * 2048)

        # --- Helper functions for un-scaling ---

        def _unscale_coords(resized_coords: List[int]) -> List[int]:
            """Converts coordinates from resized space to original space."""
            if not isinstance(resized_coords, list) or len(resized_coords) != 2:
                raise ValueError("Invalid coordinates format.")
            if resized_width == 0 or resized_height == 0:
                # Avoid division by zero; return unscaled as a fallback.
                return resized_coords

            resized_x, resized_y = resized_coords
            
            original_x = int((resized_x / 1000) * original_width)
            original_y = int((resized_y / 1000) * original_height)
            
            return [original_x, original_y]

        def _unscale_distance(resized_dist: int, direction: str) -> int:
            """Converts scroll distance from resized space to original space."""
            if direction == 'up' or direction == 'down':
                if resized_height == 0: return resized_dist
                # Vertical scroll distance scales with height
                return int((resized_dist / 1000) * original_height)
            elif direction == 'left' or direction == 'right':
                if resized_width == 0: return resized_dist
                # Horizontal scroll distance scales with width
                return int((resized_dist / 1000) * original_width)
            return resized_dist

        # --- Main action processing logic ---

        try:
            if name in ['click', 'hover', 'move_to', 'double_click']:
                resized_coords = properties['coords']
                original_coords = _unscale_coords(resized_coords)
                return f"{name} [{original_coords[0]}, {original_coords[1]}]"

            elif name == 'type':
                resized_coords = properties['coords']
                original_coords = _unscale_coords(resized_coords)
                content = properties['content']
                if 'press_enter_after' in properties:
                    return f"type [{original_coords[0]}, {original_coords[1]}] [{content}] [{properties['press_enter_after']}]"
                else:
                    return f"type [{original_coords[0]}, {original_coords[1]}] [{content}]"
            
            elif name == 'press':
                key = properties['key']
                if 'coords' in properties:
                    resized_coords = properties['coords']
                    original_coords = _unscale_coords(resized_coords)
                    return f"press [{original_coords[0]}, {original_coords[1]}] [{key}]"
                else:
                    return f"press [{key}]"
            
            elif name == 'scroll':
                direction = properties['direction']
                resized_distance = properties['distance']
                original_distance = _unscale_distance(resized_distance, direction)
                
                # Check if coords parameter exists
                if 'coords' in properties and properties['coords']:
                    resized_coords = properties['coords']
                    original_coords = _unscale_coords(resized_coords)
                    return f"scroll [{direction}] [{original_distance}] [{original_coords[0]}, {original_coords[1]}]"
                else:
                    return f"scroll [{direction}] [{original_distance}]"
            
            elif name == 'browser_select_option':
                resized_coords = properties['coords']
                original_coords = _unscale_coords(resized_coords)
                option = properties['option']
                return f"browser_select_option [{original_coords[0]}, {original_coords[1]}] [{option}]"

            # --- Actions without coordinates or distances (no changes needed) ---
            elif name == 'new_tab':
                return "new_tab"
            
            elif name == 'tab_focus':
                return f"tab_focus [{properties['tab_index']}]"
            
            elif name == 'close_tab':
                return "close_tab"
            
            elif name == 'goto':
                return f"goto [{properties['url']}]"
            
            elif name == 'go_back':
                return "go_back"
            
            elif name == 'go_forward':
                return "go_forward"

            elif name == 'wait':
                return f"wait [{properties['seconds']}]"
            
            elif name == 'find':
                search_term = properties['search_term']
                return f"find [{search_term}]"
            
            elif name == 'get_file':
                resized_coords = properties['coords']
                original_coords = _unscale_coords(resized_coords)
                return f"get_file [{original_coords[0]}, {original_coords[1]}]"

            # If the action name is not recognized
            print(f"Warning: Unrecognized action name '{name}'")
            return None

        except KeyError as e:
            print(f"Error: Missing required property {e} for action '{name}'.")
            return None
        except (ValueError, TypeError) as e:
            print(f"Error: Invalid data for action '{name}': {e}")
            return None



    def toolargs2webaction(self, tools_args, tools_env, raw_prediction):
        if OBSERVATION_TYPE == "image_som":
            action_str = self.convert_action_to_string_id(tools_args)
        else:
            action_str = self.convert_action_to_string_coords(tools_args, tools_env)
        if action_str is None:
            return None, 'None'
        if OBSERVATION_TYPE == "image_som":
            web_action = create_id_based_action(action_str)
        else:
            web_action = create_coords_based_action(action_str)
        if web_action is not None:
            web_action['raw_prediction'] = raw_prediction
        return web_action, action_str

    def execute(self, 
                args: Dict,         
                envs: ScriptBrowserEnv,) -> str:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing: 
            
        Returns:
            Formatted search results
        """
        
        try:
            action, action_str = self.toolargs2webaction(args, None, None)
            # 调用API进行搜索
            #这里替换成浏览器的action
            if action is None:
                return {"content": f"{args} action parser error", "success": False}
            resuts = envs.step(
                action
            )
            # msg = (
            #     observation,
            #     float(success),  # reward
            #     False,  # terminated
            #     False,  # truncated
            #     info,
            # )
            if resuts[1] == 1.0:
                return self._format_results(resuts)
            else:
                fail_error = resuts[-1]['fail_error']
                error_msg = f"Web Browser API returned error: {fail_error}"
                print(f"[WARNING] {error_msg}")
                return {"content": error_msg, "success": False}
        except Exception as e:
            error_msg = f"Failed to execute action: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": error_msg, "success": False}
    
    def parallel_actions(self, action_list, env_list, args_list):
        """
        并行执行 astep，并健壮地收集结果。
        它不处理 terminated 状态，而是将其忠实地传递给上层调用者。
        """
        future_to_index = {}
        for i, (action, env) in enumerate(zip(action_list, env_list)):
            if action is None: continue
            future = env.owner_actor.submit(env.astep, action)
            future_to_index[future] = i
        
        results = [None] * len(action_list)
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            env = env_list[index]
            
            try:
                # 获取 astep 的完整返回结果
                result_tuple = future.result()
                results[index] = result_tuple
                
                # 为调试添加更清晰的日志
                _, _, terminated, _, info = result_tuple
                status = "TERMINATED" if terminated else "SUCCESS"
                fail_info = info.get("fail_error", "")
                print(f"Task {index} (Config: {os.path.basename(env.config_file)}) finished with status: {status}. Info: {str(fail_info)[:120]}")

            except Exception as exc:
                # 这个 except 块处理的是 astep 中未捕获的、或 run_until_complete 本身的严重错误
                fail_error_msg = f"Unhandled exception in future execution: {exc}"
                print(f'FATAL: 任务 {index} (Config: {os.path.basename(env.config_file)}) 发生严重异常: {fail_error_msg}')
                
                # 生成一个默认的失败观察值
                fallback_obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
                
                # --- 关键修改：在这里也必须返回 terminated=True ---
                results[index] = (
                    fallback_obs,
                    0.0,
                    True, # 标记为终止，因为发生了严重错误
                    False,
                    {"page": None, "image_description": f"This is a blank image and shows nothing. This might be due to the following errors: {fail_error_msg}", "fail_error": fail_error_msg},
                )
        return results

    def draw_screen_shot(self, screenshot, text_str):   
        global global_count
        image = Image.fromarray(screenshot)
        draw = ImageDraw.Draw(image)

        # Load a TTF font with a larger size
        font_path = "./media/SourceCodePro-SemiBold.ttf"
        #font_path = "media/simhei.ttf"
        font_size, padding = 16, 2
        font = ImageFont.truetype(font_path, font_size)
        image_width, image_height = image.size
        x = (image_width) / 2
        y = (image_height) / 2
        draw.text((x, y), text_str, fill="red", font=font)
        image.save(os.path.join(os.environ.get("SCREENSHOT_SAVE_DIR", "/tmp"), f'screenshot_{global_count}.png'))
        global_count += 1
        return 
        


    def batch_execute(self, 
        args_list: List[Dict],
        envs_list: List[AsyncScriptBrowserEnv],
        raw_prediction_list: List[str]) -> List[Dict]:
        """
        Execute multiple search queries in batch
        
        Args:
            args_list: List of tool parameters
            
        Returns:
            List of formatted search results
        """
        # 提取查询和限制
        #print("args_list: ", args_list)
        action_list = []
        action_str_list = []
        for args_ind in range(len(args_list)):
            env_i = envs_list[args_ind]
            args_i = args_list[args_ind]
            raw_prediction_i = raw_prediction_list[args_ind]
            action_i, action_str_i = self.toolargs2webaction(args_i, env_i, raw_prediction_i) 
            action_list.append(action_i)
            action_str_list.append(action_str_i)

        #调整坐标
        
        if True:
        #try:
            #print("action_list: ", action_list)
            action_exe_time = time.time()
            results = self.parallel_actions(action_list, envs_list, args_list)
            # 调用批量搜索API
            output_list = []
            for res_ind, results_i in enumerate(results):
                #             return (
                # screenshot,
                # float(success),
                # False,
                # False,
                # {
                #     "page": DetachedPage(self.page.url, content),
                #     "fail_error": fail_error,
                # },
                if results_i is not None:
                    if results_i[1] == 1.0:
                        output_list.append(self._format_results(results_i, envs_list[res_ind]))
                        #self.draw_screen_shot(results_i[0], action_str_list[res_ind])
                        #print(f"Action Success: {action_str_list[res_ind]}")
                    else:
                        fail_error = results_i[-1]['fail_error']
                        error_msg = f"ERROR: Execute Web Broswer API Returned Error: "
                        if fail_error:
                            error_msg += f"{fail_error}. When Executing The Action: {action_str_list[res_ind]}"
                        output_list.append({"content": error_msg, "success": False})
                        #print(f"[ERROR] {error_msg}\n action list: {action_list[res_ind]}: \n {args_list[res_ind]}")
                else:
                    error_msg = f"ERROR: Execute Web Broswer Action Parsing Error For The Action: {args_list[res_ind]}"
                    output_list.append({"content": error_msg, "success": False})
                    #print(f"[ERROR] {error_msg}")
            return output_list
        # except Exception as e:
        #     error_msg = f"Failed to execute batch Web Browser: {str(e)}"
        #     print(f"[ERROR] {error_msg}")
        #     return [{"error": error_msg} for _ in args_list]

    def _format_results(self, api_result, curren_env) -> dict:
        """
        Format API search results for better readability
        
        Args:
            api_result: API response containing search results
            
        Returns:
            Formatted results as a string
        """
        # msg = (
        #     observation,
        #     float(success),  # reward
        #     False,  # terminated
        #     False,  # truncated
        #     info,
        # )
        # observation
        # {"text": text_obs, "image": image_obs}
        #info
        # "page": DetachedPage(self.page.url, self.page.content()),
            # "url": url,
            # "content": html
        # "fail_error": fail_error,
        # "observation_metadata": observation_metadata, 
            # "text":  {"obs_nodes_info": }
                    # "backend_id": node["backendDOMNodeId"],
                    # "union_bound": node["union_bound"],
                    # "text": node_str,
            # "image":  {"obs_nodes_info":id2center}
    
        # 单个查询的结果
        if api_result[1] == 1.0:
            pil_image = Image.fromarray(api_result[0])
            image_description = ""
            if api_result[-1]['image_description'] is not None:
                image_description = api_result[-1]['image_description']
                image_description = f"The Description of the Web Browser ScreenShot: {image_description}"

            current_page_num = 0
            for page_ind, page_i in enumerate(curren_env.context.pages):
                if page_i == curren_env.page:
                    current_page_num = page_ind
                    break
            results_dict = {
                "content": f"Step {curren_env.current_step}. This Tab Number is {current_page_num} in the browser.\n<image>\n{image_description}" , 
                "image":[pil_image],
                "success": True,
            }
            curren_env.current_step += 1
            return results_dict
        else:
            return {'content': api_result[-1]['page'].content, "success": False,}

    def save_image(self, image, image_name):
        image_pil = Image.fromarray(image)
        save_path = os.path.join(os.environ.get("DEBUG_IMAGE_SAVE_DIR", "/tmp"), f"{image_name}.png")
        image_pil.save(save_path)

    def save_last_two_images(self, env: AsyncScriptBrowserEnv, 
        save_path: str = os.environ.get("DEBUG_IMAGE_SAVE_DIR", "/tmp"),
        prefix: str = "last_two_images", coords: Tuple[int, int] = None):
        """
        Save the last two images
        """
        previous_image = None
        current_image = None
        for ind in range(len(env.trajectory)-1, -1, -1):
            if "observation" in env.trajectory[ind] and current_image is None:
                current_image = env.trajectory[ind]['observation']['image']
                continue
            if "observation" in env.trajectory[ind] and current_image is not None:
                previous_image = env.trajectory[ind]['observation']['image']
                break
        if previous_image is None and current_image is None:
            return
        if previous_image is None or current_image is None:
            return
        previous_image = Image.fromarray(previous_image)
        current_image = Image.fromarray(current_image)
        if coords is not None:
            previous_image = draw_image_with_coords(previous_image, coords)
            current_image = draw_image_with_coords(current_image, coords)
        previous_image.save(os.path.join(save_path, f"{prefix}_previous_image.png"))
        current_image.save(os.path.join(save_path, f"{prefix}_current_image.png"))
        return

    def judge_coords_on_interactable_element(self, tool_args, env: AsyncScriptBrowserEnv):
        """
        Judge if the coordinates are on an interactable element
        """
        action, action_str = self.toolargs2webaction(tool_args, env, None)
        action_type = action["action_type"]

        coord_actions_to_validate = {
            ActionTypes.MOUSE_CLICK,
            ActionTypes.DOUBLE_CLICK,
            ActionTypes.TYPE,  # Because it clicks first to focus
            ActionTypes.COORDS_SELECT_OPTION,
            ActionTypes.MOUSE_HOVER,
            ActionTypes.MOVE_TO,
        }
        if action_type in coord_actions_to_validate:
            if np.any(action["coords"]):
                x, y = action["coords"]
                # Validate that the coordinates point to an interactable element
                if env.observation_handler.is_coords_on_interactable_element(float(x), float(y)):
                    #self.save_last_two_images(env,  prefix=f"coords_on_interactable_element_uuid{uuid.uuid4()}", coords=(x, y))
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def judge_url_change(self, env: AsyncScriptBrowserEnv):
        """
        Judge if the url has changed
        """
        previous_url = None
        current_url = None
        for ind in range(len(env.trajectory)-1, -1, -1):
            if "info" in env.trajectory[ind] and current_url is None:
                if "page" in env.trajectory[ind]['info']:
                    current_url = getattr(env.trajectory[ind]['info']['page'], 'url', None)
                    continue
            if "info" in env.trajectory[ind] and current_url is not None:
                if "page" in env.trajectory[ind]['info']:
                    previous_url = getattr(env.trajectory[ind]['info']['page'], 'url', None)
                    break
        
        if previous_url is None and current_url is None:
            return False
        if previous_url is None or current_url is None:
            return True
        if previous_url != current_url:
            #self.save_last_two_images(env, prefix=f"url_change_uuid{uuid.uuid4()}")
            return True
        else:
            return False

    def judge_gain_progress_vlm(self, env: AsyncScriptBrowserEnv):
        """
        Judge if the coordinates are on an interactable element
        """

        previous_image = None
        current_image = None
        for ind in range(len(env.trajectory)-1, -1, -1):
            if "observation" in env.trajectory[ind] and current_image is None:
                current_image = env.trajectory[ind]['observation']['image']
                continue
            if "observation" in env.trajectory[ind] and current_image is not None:
                previous_image = env.trajectory[ind]['observation']['image']
                break
        if previous_image is None and current_image is None:
            return False
        if previous_image is None or current_image is None:
            return True
        results = compare_two_images_with_shangshu(previous_image, current_image, env.user_query)
        if "Final Answer: Yes".lower() in results.lower():
            #self.save_last_two_images(env,  prefix=f"gain_progress_uuid{uuid.uuid4()}")
            return True
        else:
            return False

    def judge_image_change(self, env):
        previous_image = None
        current_image = None
        for ind in range(len(env.trajectory)-1, -1, -1):
            if "observation" in env.trajectory[ind] and current_image is None:
                current_image = env.trajectory[ind]['observation']['image']
                continue
            if "observation" in env.trajectory[ind] and current_image is not None:
                previous_image = env.trajectory[ind]['observation']['image']
                break
        if previous_image is None and current_image is None:
            return False
        if previous_image is None or current_image is None:
            return True
        ssim = calculate_ssim(previous_image, current_image)
        if ssim == 1.0:
            return False
        else:
            #self.save_last_two_images(env,  prefix=f"image_change_uuid{uuid.uuid4()}")
            return True

    async def calculate_reward(self, args: Dict, result: Dict, env: AsyncScriptBrowserEnv, penalty_reward=-0.001) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        #print(f"result: {result}")
        result_obj = result
        # 有效的工具调用
        if "success" in result_obj:
            if result_obj["success"]:
                if 'wPRALL'.lower() in EXPERIMENT_NAME.lower():
                    #惩罚主要基于页面变化，页面没有变化时，不惩罚
                    if self.judge_url_change(env):
                        #url 变了页面肯定变了
                        return 0.0
                    elif self.judge_coords_on_interactable_element(args, env):
                        #点了可交互元素，有效
                        return 0.0
                    elif not self.judge_image_change(env):
                        #页面没有变化，惩罚
                        return penalty_reward
                    elif self.judge_gain_progress_vlm(env):
                        #页面有变化，有效
                        return 0.0
                    else:
                        return penalty_reward
                else:
                    return 0.0
            else:
                return penalty_reward
        else:
            return penalty_reward
