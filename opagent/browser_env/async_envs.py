import asyncio
import json
from pathlib import Path
from typing import Any
import numpy as np
from numpy.random import rand
import numpy.typing as npt
from beartype import beartype
from gymnasium import Env
from gymnasium.spaces import Box, Text
from playwright.async_api import Page, ViewportSize, async_playwright
from playwright.async_api import Playwright, Browser, PlaywrightContextManager, BrowserContext
import playwright
import nest_asyncio
from PIL import Image

from .actions import Action, aexecute_action, get_action_space, create_id_based_action, aexecute_action_coords
from .processors import VLM_EXP_DEBUG, ObservationHandler, ObservationMetadata, AsyncImageObservationProcessor
from .utils import DetachedPage, png_bytes_to_numpy, Observation, StateInfo
from .trajectory import Trajectory
import os
from opagent.evaluation_harness.asyns_evaluators import evaluator_router, image_utils, EvaluatorComb

action_mapper = None
class Executor: pass
import urllib.parse
import re
from typing import List, Set, Optional, Union
from datetime import datetime
from playwright.async_api import Error as PlaywrightError
import logging
from .utils import with_timeout_legacy, change_mainip2ecsip
import socket
import random
import glob
import threading
import concurrent.futures
from dataclasses import dataclass, field
from demo.evaluation_harness.webjudge.src.qwen_vlm_request import describe_status_image_with_shangshu
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

WEBARENA_AUTH_PATH = os.environ.get("WEBARENA_AUTH_PATH", None)
OBSERVATION_TYPE = os.environ.get('OBSERVATION_TYPE', '')
ACTION_EXECUTE_MODE = os.environ.get("ACTION_EXECUTE_MODE", "playwright_api")
GPUS_PER_NODE = os.environ.get("GPUS_PER_NODE", 1)
EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', "")
WEBARENA_PROXY = os.environ.get("WEBARENA_PROXY", None)
SAVE_MODEL_PATH = os.environ.get("SAVE_MODEL_PATH", None)
NUM_NODES = os.environ.get("NUM_NODES", 1)
TENSORBOARD_DIR = os.environ.get("TENSORBOARD_DIR")
TASK_ID = os.environ.get("TASK_ID")
BROWSER_OUTPUT_PATH = TENSORBOARD_DIR.replace("tensorboard", "browser_config")
BROWSER_OUTPUT_PATH = BROWSER_OUTPUT_PATH + "/" + TASK_ID + "/"
print("BROWSER_OUTPUT_PATH: ", BROWSER_OUTPUT_PATH)
global_count = 0

def get_ws_endpoint_list():
    # [MODIFIED] - 全局 ws_endpoint_list 在模块加载时准备好
    ws_endpoint_list = []
    try:
        browser_config_files = []
        #每个节点都有浏览器
        while len(browser_config_files) < int(NUM_NODES):
            browser_config_dir = BROWSER_OUTPUT_PATH
            browser_config_files = glob.glob(os.path.join(browser_config_dir, "*.json"))
            logger.info(f"Found {len(browser_config_files)} browser config files in {browser_config_files} {browser_config_dir}.")
        for file_path in browser_config_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded browser config from {file_path}: {data}")
            ws_endpoint_list.extend(data)
        logger.info(f"Loaded {len(ws_endpoint_list)} browser WebSocket endpoints.")
    except Exception as e:
        logger.error(f"Failed to load browser WebSocket endpoints: {e}", exc_info=True)
    return ws_endpoint_list


# [MODIFIED] - BrowserUnit 数据结构保持不变，但其 loop 将被 Actor 严格管理
@dataclass
class BrowserUnit:
    endpoint: str
    playwright_manager: PlaywrightContextManager
    playwright: Playwright
    browser: Browser
    loop: asyncio.AbstractEventLoop
    # lock 不再需要，因为 Actor 已经提供了串行访问
    # lock: threading.Lock = field(default_factory=threading.Lock)


# [NEW] - BrowserActor 类，这是新架构的核心
class BrowserActor:
    """
    将一个 BrowserUnit 及其事件循环封装在一个专用的后台线程中。
    所有与该浏览器单元的异步交互都通过线程安全的方法提交。
    """
    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.browser_unit: Optional[BrowserUnit] = None
        self._start_event = threading.Event()

    def start(self):
        """启动后台线程和事件循环。"""
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run, name=f"BrowserActor-{self._endpoint.split(':')[-1]}")
        self._thread.daemon = True
        self._thread.start()
        
        # 等待线程内部的 loop 和 browser_unit 准备就绪
        # 添加超时以防死锁
        if not self._start_event.wait(timeout=60.0):
             raise RuntimeError(f"BrowserActor for {self._endpoint} timed out during startup.")
        
        if not self.browser_unit:
            raise RuntimeError(f"BrowserActor for {self._endpoint} failed to initialize its BrowserUnit.")

    def _run(self):
        """线程的目标函数：创建循环，运行它，并处理任务。"""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            self.browser_unit = self._loop.run_until_complete(
                _create_browser_unit_async(self._endpoint, self._loop)
            )
            
            #if self.browser_unit:
                #logger.info(f"BrowserActor for {self._endpoint} initialized successfully in thread {threading.get_ident()}.")
            if not self.browser_unit:
                logger.error(f"Failed to initialize BrowserUnit in BrowserActor for {self._endpoint}.")

        except Exception as e:
            logger.error(f"Exception during BrowserActor initialization: {e}", exc_info=True)
            self.browser_unit = None
        finally:
            self._start_event.set()

        if self.browser_unit and self._loop:
            self._loop.run_forever()

        if self._loop:
            self._loop.close()
        #logger.info(f"Event loop for BrowserActor {self._endpoint} has been closed.")

    def submit(self, coro_func, *args, **kwargs) -> concurrent.futures.Future:
        """
        从外部线程安全地提交一个协程函数到此 Actor 的事件循环中执行。
        返回一个 concurrent.futures.Future 对象，可以用来获取结果。
        """
        if not self._loop or not self._loop.is_running():
            raise RuntimeError(f"BrowserActor for {self._endpoint} is not running.")
        
        async def coro_wrapper():
            return await coro_func(*args, **kwargs)

        return asyncio.run_coroutine_threadsafe(coro_wrapper(), self._loop)

    def stop(self):
        """停止事件循环和线程。"""
        if not self._loop or not self._loop.is_running():
            return

        if self.browser_unit:
            try:
                # 提交关闭任务，但不等待，因为我们即将停止整个循环
                self.submit(self.browser_unit.browser.close).result(timeout=15)
                self.submit(self.browser_unit.playwright.stop).result(timeout=10)
            except Exception as e:
                logger.error(f"Error during graceful shutdown of browser resources for {self._endpoint}: {e}")

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=10)
        
        #logger.info(f"BrowserActor for {self._endpoint} stopped.")

# [MODIFIED] - _create_browser_unit_async 函数现在使用全局锁
PLAYWRIGHT_STARTUP_LOCK = threading.Lock()
async def _create_browser_unit_async(endpoint: str, loop: asyncio.AbstractEventLoop) -> Optional[BrowserUnit]:
    """连接到现有的浏览器并创建一个 BrowserUnit 实例。"""
    playwright_manager = async_playwright()
    playwright = None
    
    # ------------------- 关键修改开始 -------------------
    # 在进入异步上下文之前，我们先获取同步的线程锁
    with PLAYWRIGHT_STARTUP_LOCK:
        #logger.info(f"Thread {threading.get_ident()} acquired Playwright startup lock for {endpoint}.")
        try:
            # 只有获取到锁的线程才能执行 start()
            # start() 内部会创建子进程，现在这个过程是串行的
            playwright = await playwright_manager.start()
            #logger.info(f"Playwright driver started successfully for {endpoint}.")
        except Exception as e:
            logger.error(f"Failed to start Playwright driver for {endpoint}: {e}", exc_info=True)
            # 即使失败也要确保 playwright_manager 被正确处理，尽管这里它可能没有完全启动
            # Playwright 的 __aexit__ 应该能处理这种情况
            try:
                await playwright_manager.stop()
            except Exception:
                pass
            return None
        #finally:
            #logger.info(f"Thread {threading.get_ident()} released Playwright startup lock for {endpoint}.")
    # ------------------- 关键修改结束 -------------------
    
    # 一旦驱动进程启动，后续的连接操作可以完全并行，无需锁
    try:
        browser = await playwright.chromium.connect_over_cdp(endpoint)
        #logger.info(f"Successfully connected to browser at {endpoint}")
        return BrowserUnit(
            endpoint=endpoint,
            playwright_manager=playwright_manager, # 保存 manager 以便后续关闭
            playwright=playwright,
            browser=browser,
            loop=loop
        )
    except Exception as e:
        logger.error(f"Failed to connect to browser at {endpoint} after starting driver: {e}", exc_info=True)
        # 如果连接失败，需要停止刚刚启动的 playwright 实例
        if playwright:
            await playwright.stop()
        return None

# [MODIFIED] - 新的初始化函数，创建 Actor 池
def initialize_browser_actor_pool(webbrowser_thread_executor: concurrent.futures.ThreadPoolExecutor) -> List[BrowserActor]:
    """初始化并启动 BrowserActor 池。"""
    ws_endpoint_list = get_ws_endpoint_list()
    if not ws_endpoint_list:
        raise RuntimeError("No WebSocket endpoints found for browser initialization.")

    actors = [BrowserActor(ep) for ep in ws_endpoint_list]
    successful_actors = []
    
    future_to_actor = {webbrowser_thread_executor.submit(actor.start): actor for actor in actors}
    for future in concurrent.futures.as_completed(future_to_actor):
        actor = future_to_actor[future]
        try:
            future.result()
            if actor.browser_unit:
                successful_actors.append(actor)
            else:
                logger.error(f"Actor for {actor._endpoint} started but failed to initialize its browser unit.")
        except Exception as exc:
            logger.error(f"Actor for {actor._endpoint} failed to start: {exc}", exc_info=True)

    if not successful_actors:
        raise RuntimeError("Browser actor pool initialization failed completely.")
    
    #logger.info(f"Successfully initialized {len(successful_actors)} browser actors.")
    return successful_actors

def _is_endpoint_available(cdp_endpoint: str, timeout: float = 1.0) -> bool:
    try:
        parsed_url = urllib.parse.urlparse(cdp_endpoint)
        hostname = parsed_url.hostname
        port = parsed_url.port
        if not hostname or not port:
            return False
        with socket.create_connection((hostname, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError, TypeError):
        return False


def get_available_cdp_endpoint_concise(ws_endpoint_list: List[str]) -> Optional[str]:
    if not ws_endpoint_list:
        raise RuntimeError("ws_endpoint_list is empty. No remote browsers available.")

    shuffled_list = random.sample(ws_endpoint_list, k=len(ws_endpoint_list))
    return next((ep for ep in shuffled_list if _is_endpoint_available(ep)), None)

def parse_compound_url_string(compound_string: Optional[str]) -> List[str]:
    """
    解析一个特殊的复合字符串，该字符串包含一个基础URL、一个'####'分隔符
    以及一个经过URL编码的HTML片段，并从中提取所有URL到一个列表中。

    Args:
        compound_string: 要解析的字符串。
            例如: 'https://www.unjs.com/####%3Ca%20...%3E'

    Returns:
        一个包含从字符串中找到的所有URL的列表。
        如果输入无效或无法解析，则返回一个空列表。
    """
    # 1. 处理边缘情况：输入为空或类型不正确
    if not compound_string or not isinstance(compound_string, str):
        return []

    extracted_urls = []
    try:
        # 2. 对整个字符串进行URL解码
        # 这会将 '%3C' 变为 '<', '%22' 变为 '"' 等
        decoded_string = urllib.parse.unquote(compound_string)

        # 3. 按 '####' 分隔符分割字符串
        parts = decoded_string.split('####')

        # 4. 检查分割是否成功
        if len(parts) < 2:
            # 如果没有分隔符，可能整个字符串就是一个URL，直接返回
            # 否则，无法处理，返回空列表
            return [decoded_string] if decoded_string.startswith(('http://', 'https://')) else []

        # 5. 第一个部分是基础URL
        base_url = parts[0]
        if base_url:  # 确保它不为空
            extracted_urls.append(base_url)

        # 6. 第二个部分是HTML片段
        html_part = parts[1]

        # 7. 使用正则表达式从HTML片段中找到 href 属性里的URL
        # r'href="([^"]+)"' 的意思是：
        # - href="   : 匹配字面上的 href="
        # - ([^"]+)  : 捕获一个或多个不是双引号(")的字符
        # - "        : 匹配字面上的 "
        match = re.search(r'href="([^"]+)"', html_part)
        if match:
            # match.group(1) 获取第一个捕获组的内容，也就是我们需要的URL
            href_url = match.group(1)
            extracted_urls.append(href_url)

    except Exception as e:
        print(f"解析字符串时发生错误: {e}")
        # 如果在任何步骤发生意外错误，返回一个空列表以保证安全
        return []

    return extracted_urls


class AsyncScriptBrowserEnv(Env[npt.NDArray[np.uint8], Action]):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """

    def __init__(
            self,
            context_id: str,
            max_page_length: int = 2048,
            headless: bool = True,
            slow_mo: int = 0,
            timeout: int = 30000,
            viewport_size: ViewportSize = {"width": 1280, "height": 720},
            observation_type: str = "image_som",
            current_viewport_only: bool = False,
            save_trace_enabled: bool = False,
            sleep_after_execution: float = 0.0,
            captioning_fn=None,
            user_query=None,
    ):
        # self.observation_space = Box(
        #     0,
        #     255,
        #     (viewport_size["height"], viewport_size["width"], 4),
        #     np.uint8,
        # )
        # TODO: make Space[Action] = ActionSpace
        self.context_id = context_id
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.reset_finished = False
        self.timeout = timeout
        self.viewport_size = viewport_size

        self.current_viewport_only = current_viewport_only
        self.user_query = user_query
        self.observation_handler = AsyncImageObservationProcessor(
            observation_type,
            self.viewport_size,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )

        self.trajectory: Trajectory = []
        self.evaluator: EvaluatorComb = None
        self.config_file: Path = None
        self.page: Page = None
        self.web_executor: Executor = None
        self.current_step = 1
        
        # 新增属性来存储训练step信息
        self.training_step = 0
        self.trajectory_save_freq = 20
        self.trajectory_save_enabled = False
        self.epoch = 0

        self.browser_unit: Optional[BrowserUnit] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.reset_finished = False

    def save_image(self, bbox_img_np, save_name):
        now = datetime.now()
        timestamp_str = ""
        # timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        # timestamp_str = str(uuid.uuid4())
        # TODO: 配置您的临时文件路径
        tmp_dir = os.getenv("TMP_DIR", "./tmp")
        self.save_img_file = f"{tmp_dir}/{save_name}_{timestamp_str}_{self.context_id}.png"
        logger.info(f"Saving bounding box image to: {self.save_img_file}")
        bbox_img = Image.fromarray(bbox_img_np)
        bbox_img.save(self.save_img_file)
        return

    async def setup(self, config_file: Optional[Path] = None, *, storage_state=None, start_url=None, options=None) -> None:

        # 如果有传入 storage_state，则直接使用

        if storage_state is not None:
            # logger.info(f"INFO [Env/{self.context_id}]: Adding storage_state. config_file: {config_file}")
            # logger.info(f"INFO [Env/{self.context_id}]: storage_state: {storage_state}")
            self.context = await self.browser_unit.browser.new_context(
                proxy={"server": WEBARENA_PROXY} if WEBARENA_PROXY else None,
                viewport=self.viewport_size,
                storage_state=storage_state,
                device_scale_factor=1,
                ignore_https_errors=True,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            )
        elif config_file:
            #训练任务应该都会走到这里
            with open(config_file, "r") as f:
                instance_config = json.load(f)

            start_url = instance_config.get("start_url", None)
            if "REPLACE_WITH_YOUR_HOST" in options and options["REPLACE_WITH_YOUR_HOST"] is not None:
                start_url = change_mainip2ecsip(start_url, options["REPLACE_WITH_YOUR_HOST"])
            # Use custom viewport size if specified in the config, otherwise use the default.
            viewport_size = self.viewport_size.copy()
            viewport_size.update(instance_config.get("viewport_size", {}))
            self.observation_handler.viewport_size = viewport_size
            #storage_state = instance_config.get("storage_state", None)
            if options is not None and "storage_state" in options:
                #storage_state 是从tbase获取的
                storage_state = options["storage_state"]
            storage_state_cookies = None
            if storage_state is not None:
                storage_state_cookies = storage_state.get("cookies", None)
            if storage_state_cookies is None or len(storage_state_cookies) == 0:
                #cookie为空，则使用config的storage_state
                config_storage_state = instance_config.get("storage_state", None)
                if config_storage_state is not None:
                    if config_storage_state.startswith("./"):
                        if "REPLACE_WITH_YOUR_HOST" in options and options["REPLACE_WITH_YOUR_HOST"] is not None:
                            storage_state = os.path.join(WEBARENA_AUTH_PATH+f"_{options['REPLACE_WITH_YOUR_HOST']}"+"/", config_storage_state[2:])
                        else:
                            storage_state = os.path.join(WEBARENA_AUTH_PATH, config_storage_state[2:])
            #print(f"INFO [Env/{self.context_id}]: storage_state: {storage_state}")
            geolocation = instance_config.get("geolocation", None)
            self.context = await self.browser_unit.browser.new_context(
                proxy={"server": WEBARENA_PROXY} if WEBARENA_PROXY else None,
                viewport=self.viewport_size,
                storage_state=storage_state,
                geolocation=geolocation,
                device_scale_factor=1,
                ignore_https_errors=True,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            )
        else:
            if options is not None and "storage_state" in options:
                storage_state = options["storage_state"]
            self.context = await self.browser_unit.browser.new_context(
                proxy={"server": WEBARENA_PROXY} if WEBARENA_PROXY else None,
                viewport=self.viewport_size,
                device_scale_factor=1,
                storage_state=storage_state,
                ignore_https_errors=True,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            )
        self.page = await self.context.new_page()
        if ACTION_EXECUTE_MODE == "playwright_cdp":
            # 初始化web_executor
            self.web_executor = Executor(self.page)
            await self.web_executor.new_cdp_session()

        if "|AND|" in start_url:
            start_url = start_url.split("|AND|")[0].strip()
        # 解析start_url，start_url可能包含多个网站只取第一个
        start_url = parse_compound_url_string(start_url)[0]
        #logger.info(f"INFO [Env/{self.context_id}]: start_url: {start_url}")
        if start_url:
            if VLM_EXP_DEBUG == '1':
                max_retries = 3
                timeout = 1 * 60 * 1000
                retry_delay = 3
            else:
                max_retries = 10
                timeout = 3 * 60 * 1000
                retry_delay = 10
            last_exception = None  # 用于存储最后一次的异常

            for attempt in range(max_retries):
                try:
                    # 尝试访问页面
                    await self.page.goto(start_url, wait_until="networkidle", timeout=timeout)

                    # 如果成功，打印信息并跳出循环
                    # print(f"Successfully navigated to {start_url} on attempt {attempt + 1}.")
                    last_exception = None  # 成功后清除异常记录
                    break

                except Exception as e:
                    # 捕获异常，记录下来
                    last_exception = e
                    print(f"Attempt {attempt + 1}/{max_retries} failed for URL: {start_url}. Error: {e}")

                    if attempt < max_retries - 1:
                        # 等待后重试

                        print(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                    else:
                        # 所有重试都用完了
                        # 尝试访问页面
                        print("All retry attempts failed. goto www.baidu.com")
                        await self.page.goto("https://www.baidu.com", wait_until="networkidle", timeout=timeout)
                        last_exception = None  # 成功后清除异常记录
                        break

            # 在循环结束后，检查是否仍然存在未解决的异常
            if last_exception:
                # 如果有，将它重新抛出，以便上层代码可以捕获这个最终的失败
                raise last_exception

    @beartype
    async def areset(
            self,
            *,
            seed: Optional[int] = None,
            browser_unit: BrowserUnit,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[npt.NDArray[np.uint8], dict[str, object]]:
        """
        Reset the environment with a robust retry and fallback mechanism.
        The entire browser interaction sequence (setup, wait, get content/screenshot)
        is wrapped in a retry loop. If it fails, it will fall back to loading www.baidu.com.

        :param options: options for the environment. The options are:
            - storage_state: the path to the storage state file
            - start_url: the initial URL to navigate to
            - config_file: path to the configuration file for the task
        """
        # --- 常量定义 ---
        MAX_RETRIES = 5
        RETRY_DELAY_SECONDS = 3
        DEFAULT_FALLBACK_URL = "https://www.baidu.com"
        #logger.info(f"INFO [Env/{self.context_id}]: in areset...")
        # --- 步骤 1: 初始状态重置和不可重试的配置检查 ---
        super().reset(seed=seed, options=options)
        if self.reset_finished:
            await self.aclose()

        if self.reset_finished:
            # 如果之前有 context，确保关闭
            if self.context and not self.context.is_closed():
                await self.context.close()

        self.browser_unit = browser_unit


        config_file = ""
        # 检查config_file是否存在是配置错误，不应该重试，所以提前处理
        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if not config_file.exists():
                raise ValueError(f"Config state {config_file} does not exist. This is a fatal configuration error.")

        # --- 步骤 2: 带重试的核心页面加载和观察获取逻辑 ---
        load_success = False
        screenshot, content = self._get_fallback_observation("Initial Observation")
        for attempt in range(MAX_RETRIES):
            try:
                # 每次重试都从 setup 开始
                if options is not None and "config_file" in options:
                    await self.setup(config_file=config_file, options=options)
                elif options is not None and "storage_state" in options and "start_url" in options:
                    await self.setup(storage_state=options['storage_state'], start_url=options['start_url'], options=options)
                elif options is not None and "start_url" in options:
                    await self.setup(start_url=options['start_url'], options=options)
                else:
                    await self.setup() # 默认 setup

                # 等待页面加载完成
                await self.page.wait_for_load_state("networkidle", timeout=30000)
                
                # 现在将内容和截图获取也包含在try块中
                content = await self.page.content()
                screenshot, content = await self._async_get_obs() # 假设这个函数也可能更新content

                #logger.info(f"INFO [Env/{self.context_id}]: Successfully loaded and observed page on attempt {attempt + 1}.")
                load_success = True
                break  # 所有操作成功，跳出重试循环

            except (PlaywrightError, TimeoutError, Exception) as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed during browser interaction. Error: {e}. "
                    f"Retrying in {RETRY_DELAY_SECONDS} seconds..."
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
        
        # --- 步骤 3: 如果所有重试都失败，执行回退操作 ---
        if not load_success:
            logger.error(f"All {MAX_RETRIES} attempts failed. Falling back to {DEFAULT_FALLBACK_URL}.")
            try:
                # 检查 self.page 是否为 None，如果是则重新创建
                if self.page is None:
                    logger.warning("self.page is None, attempting to recreate browser context...")
                    # 重新创建浏览器上下文和页面                 
                    if not hasattr(self, 'context') or self.context is None:
                        # 重新创建上下文
                        self.context = await self.browser_unit.browser.new_context(
                            viewport=self.viewport_size,
                            device_scale_factor=1,
                            ignore_https_errors=True,
                            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
                        )
                    
                    # 重新创建页面
                    self.page = await self.context.new_page()
                    if ACTION_EXECUTE_MODE == "playwright_cdp":
                        # 重新初始化web_executor
                        self.web_executor = Executor(self.page)
                        await self.web_executor.new_cdp_session()
                
                # 现在在 page 对象上导航到回退URL
                await self.page.goto(DEFAULT_FALLBACK_URL, wait_until="networkidle", timeout=30000)
                
                # 在回退成功后，再次获取内容和截图
                content = await self.page.content()
                screenshot, content = await self._async_get_obs()
                #logger.info("Successfully loaded and observed fallback URL.")
            except (PlaywrightError, TimeoutError, Exception) as fallback_e:
                # 如果连回退页面的观察都失败了，这是一个严重问题
                error_message = f"FATAL: Fallback to {DEFAULT_FALLBACK_URL} also failed. The environment is unrecoverable. {fallback_e}"
                logger.error(error_message, exc_info=True)  # exc_info=True 会记录完整的异常堆栈

        # --- 步骤 4: 无论成功还是回退，都执行后续的通用逻辑 ---
        if self.config_file is None:
            self.config_file = config_file
        self.reset_finished = True

        # 此时 screenshot 和 content 应该已经被成功赋值
        # （要么来自主逻辑，要么来自回退逻辑）
        observation = {"text": content, "image": screenshot}
        state_info: StateInfo = {"observation": observation, "info": {"page": DetachedPage(self.page.url, content)}}
        self.trajectory.append(state_info)
        
        if self.config_file:
                self.evaluator = evaluator_router(
                self.config_file, captioning_fn=None, REPLACE_WITH_YOUR_HOST=options['REPLACE_WITH_YOUR_HOST'] if options is not None and 'REPLACE_WITH_YOUR_HOST' in options else None
            )

        return (
            screenshot,
            {"page": DetachedPage(self.page.url, content)},
        )

    async def get_score(self, solution_str):
        score = await self.evaluator(
            solution_str=solution_str,
            trajectory=self.trajectory,
            config_file=self.config_file,
            page=self.page,
        )
        return score

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, str]] = None,
    ) -> tuple[npt.NDArray[np.uint8], dict[str, object]]:
        self.current_step = 1 
        return asyncio.run(self.areset(seed=seed, options=options))

    @with_timeout_legacy(2 * 60)
    # [MODIFIED] aclose 现在只关闭 context
    async def aclose(self):
        """
        仅关闭此环境实例持有的 BrowserContext。
        BrowserUnit 的生命周期由外部的 Actor 管理。
        """
        if not self.reset_finished:
            return

        # 检查 context 是否存在
        if self.context:
            try:
                # 直接调用 close()。Playwright 的 close() 是幂等的，
                # 对已关闭的 context 调用不会产生错误。
                await self.context.close()
                #logger.info(f"INFO [Env/{self.context_id}]: BrowserContext closed (or was already closed).")
            except Exception as e:
                # 捕获任何意外的异常，例如在关闭过程中网络连接断开等
                # 即使关闭 context 失败，也要继续执行后续的清理步骤
                logger.warning(f"Warning [Env/{self.context_id}]: An unexpected error occurred while closing BrowserContext: {e}")
        
        # 重置实例的状态变量，为下一次 areset 做准备
        self.context = None
        self.page = None
        self.browser_unit = None # 解除对 BrowserUnit 的引用
        self.owner_actor = None # [IMPORTANT] 解除对 Actor 的引用
        self.reset_finished = False
        self.current_step = 1 # 如果适用，重置步数

        #logger.info(f"INFO [Env/{self.context_id}]: aclose finished.")
        

    def close(self) -> None:
        self.current_step = 1 
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，创建一个任务
            if loop.is_running():
                task = loop.create_task(self.aclose())
                # 等待任务完成，但设置超时
                try:
                    loop.run_until_complete(asyncio.wait_for(task, timeout=120))
                except asyncio.TimeoutError:
                    logger.warning(f"Warning: aclose() timed out for context {self.context_id}")
                    task.cancel()
            else:
                # 没有运行的事件循环，直接运行
                asyncio.run(self.aclose())
        except RuntimeError:
            # 没有事件循环，创建新的
            asyncio.run(self.aclose())
        except Exception as e:
            logger.error(f"Error during close() for context {self.context_id}: {e}")


    async def _async_get_obs(self) -> tuple[npt.NDArray[np.uint8], str]:
        obs, content_str = await self.observation_handler.async_get_observation(self.page)
        return obs, content_str

    def _get_obs_metadata(self) -> ObservationMetadata:
        metadata = self.observation_handler.get_observation_metadata()
        return metadata

    def _get_fallback_observation(self, fail_error: str) -> tuple[np.ndarray, str]:
        """辅助函数，用于在失败时生成一个默认的观察值。"""
        fallback_img = np.zeros((self.viewport_size["height"], self.viewport_size["width"], 3), dtype=np.uint8)
        fallback_content = f"Error: {fail_error}"
        return fallback_img, fallback_content

    @with_timeout_legacy(3 * 60)
    async def astep(
            self, action: Action
    ) -> tuple[npt.NDArray[np.uint8], float, bool, bool, dict[str, object]]:
        self.trajectory.append(action)
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        fail_error = ""
        success = False

        # --- 1. 入口存活检查 ---
        if self.page.is_closed():
            fail_error = "Browser/Page is not connected or has been closed before step execution."
            print(f"[astep] {fail_error}")
            screenshot, content = self._get_fallback_observation(fail_error)
            page_url_on_fail = "N/A (Browser/Page Closed)"
            return (
            screenshot, 0.0, True, False, 
            {"page": DetachedPage(page_url_on_fail, content), 
            "fail_error": fail_error, 
            "image_description": f"This is a blank image and shows nothing. This might be due to the following errors: {fail_error}"})

        # --- 2. 动作执行阶段 ---
        try:
            # 你的动作执行逻辑
            if ACTION_EXECUTE_MODE == "playwright_api":
                if OBSERVATION_TYPE == 'image_som':
                    self.page = await aexecute_action(action, self.page, self.context, self.observation_handler)
                else:
                    self.page = await aexecute_action_coords(action, self.page, self.context, self.observation_handler)
            if ACTION_EXECUTE_MODE == "playwright_cdp":
                # print("采用Playwright CDP命令方式执行action")
                # 执行动作空间翻译
                trans_action = action_mapper(action, self.observation_handler, self.page)
                # 执行翻译和动作执行
                await self.web_executor.execute(trans_action)
            success = True
        except Exception as e:
            fail_error = f"Step Action failed, Because: {e}"
            print(f"[astep] {fail_error}")
            screenshot, content = self._get_fallback_observation(fail_error)
            is_terminated = "closed" in str(e).lower()
            page_url_on_fail = self.page.url if not self.page.is_closed() else "N/A (Page Closed)"
            return (screenshot, 0.0, is_terminated, False,
                    {
                        "page": DetachedPage(page_url_on_fail, content), 
                        "fail_error": fail_error, 
                        "image_description": f"This is a blank image and shows nothing. This might be due to the following errors: {fail_error}"})

        # --- 3. 观察获取阶段 ---
        try:
            #print(f"before get obs")
            screenshot, content = await self._async_get_obs()
            #print(f"after get obs")
            if content.startswith("Error:"):
                raise RuntimeError(content)
        except Exception as e:
            fail_error = f"Getting observation failed. Because: {e}"
            print(f"[astep] {fail_error}")
            screenshot, content = self._get_fallback_observation(fail_error)
            is_terminated = "closed" in str(e).lower()
            page_url_on_fail = self.page.url if not self.page.is_closed() else "N/A (Page Closed)"
            return (screenshot, 0.0, is_terminated, False,
                    {"page": DetachedPage(page_url_on_fail, content), "fail_error": fail_error, 
                    "image_description": f"This is a blank image and shows nothing. This might be due to the following errors: {fail_error}"})

        # --- 4. 正常成功返回 ---
        observation = {"text": content, "image": screenshot}
        detached_page = DetachedPage(self.page.url, content)
        state_info: StateInfo = {"observation": observation, "info": {"page": detached_page}}
        self.trajectory.append(state_info)

        print(f"return astep [success] {self.current_step}")
        image_description = None
        if "wCap".lower() in EXPERIMENT_NAME.lower() and self.current_step > 2:
            try:
                image_description = describe_status_image_with_shangshu(Image.fromarray(screenshot), 
                        user_query=self.user_query,
                        max_tokens=150)
            except Exception as e:
                logger.warning(f"Warning: Failed to describe the Web Browser ScreenShot. Error: {e}")
                image_description = f"Cannot get the description of the Web Browser ScreenShot"
        return (
            screenshot, float(success), False, False,
            {"page": detached_page, "fail_error": fail_error, "image_description": image_description},
        )

    def step(
            self, action: Action
    ) -> tuple[npt.NDArray[np.uint8], float, bool, bool, dict[str, object]]:
        return asyncio.run(self.astep(action))
