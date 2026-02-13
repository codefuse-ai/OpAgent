import pytest
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Playwright,
)
from unittest.mock import patch
import threading
from typing import Any, Optional, Dict, Tuple, List
import socket
import requests

class PlaywrightManager:
    """
    管理共享的 Playwright 浏览器进程。
    每个 context_id 对应一个独立的浏览器进程。
    通过一个类级别的锁来确保进程启动的线程安全。
    """
    _init_lock = threading.Lock()
    # 存储结构: { context_id: {"browser_process_handler": Browser, "port": int, "cdp_endpoint": str} }
    _context_map: Dict[str, Dict[str, Any]] = {}

    _headless: bool = True
    _slow_mo: int = 0
    _timeout: int = 30000
    _ip: str = socket.gethostbyname(socket.gethostname())

    def __init__(self, headless: bool = True, slow_mo: int = 0, timeout: int = 30000):
        cls = self.__class__
        cls._headless = headless
        cls._slow_mo = slow_mo
        cls._timeout = timeout
        cls._ip = socket.gethostbyname(socket.gethostname())

    @classmethod
    def _find_free_port(cls) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    @classmethod
    def _is_browser_responsive(cls, port: int) -> bool:
        try:
            with socket.create_connection((cls._ip, port), timeout=0.5):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False

    @classmethod
    def get_websocket_url(cls, port: int) -> str:
        """
        查询HTTP端点以获取WebSocket调试URL。
        """
        try:
            # response = requests.get(f"http://{cls._ip}:{port}/json/version")
            response = requests.get(f"http://127.0.0.1:{port}/json/version")
            response.raise_for_status()
            return response.json()["webSocketDebuggerUrl"]
        except requests.exceptions.RequestException as e:
            print(f"错误：无法连接到调试端口 {port}。")
            print(f"请确认你已经使用 '--remote-debugging-port={port}' 参数启动了浏览器。")
            raise ConnectionError(f"无法连接到调试端口 {port}。") from e
        except (KeyError, IndexError):
            raise ValueError("在JSON响应中找不到'webSocketDebuggerUrl'。")

    @classmethod
    async def _ensure_browser_is_running(cls, browser_id: str):
        """
        确保与 browser_id 关联的浏览器进程正在运行。如果不在，则启动一个。
        此方法是线程安全的。
        """
        with cls._init_lock:
            if browser_id in cls._context_map:
                port = cls._context_map[browser_id]['port']
                if cls._is_browser_responsive(port):
                    return  # 浏览器已在运行

                print(
                    f"WARNING [PlaywrightManager]: Browser for '{browser_id}' on port {port} is unresponsive. Relaunching.")
                # 清理旧的、无响应的浏览器信息
                old_handler = cls._context_map.pop(browser_id, {}).get("browser_process_handler")
                if old_handler and old_handler.is_connected():
                    # Best-effort close
                    try:
                        await old_handler.close()
                    except Exception:
                        pass

            playwright_cm = async_playwright()
            playwright_instance = await playwright_cm.__aenter__()
            port = cls._find_free_port()

            print(
                f"INFO [PlaywrightManager/Thread-{threading.get_ident()}]: No active browser for '{browser_id}'. Launching a new instance on port {port}.")

            # executable_path="/usr/bin/google-chrome"
            executable_path = "/root/.cache/ms-playwright/chromium-1055/chrome-linux/chrome-wrapper"
            launch_options = {
                "executable_path": executable_path,
                "headless": cls._headless,
                "slow_mo": cls._slow_mo,
                "args": [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-zygote',
                    f'--remote-debugging-port={port}',
                ]
            }

            try:
                # `browser_process_handler` 是控制浏览器进程的对象，需要保存下来用于最终关闭
                browser_process_handler = await playwright_instance.chromium.launch(**launch_options)
            except Exception as e:
                print(f"ERROR [PlaywrightManager]: Failed to launch browser on port {port}. Error: {e}")
                raise

            cdp_endpoint = cls.get_websocket_url(port)
            # cdp_endpoint = f"http://{cls._ip}:{port}"
            cls._context_map[browser_id] = {
                "browser_process_handler": browser_process_handler,
                "port": port,
                "cdp_endpoint": cdp_endpoint,
                "ref_count": 0
            }

            print(
                f"INFO [PlaywrightManager]: Browser for '{browser_id}' launched successfully. CDP Endpoint: {cdp_endpoint}")

    @classmethod
    async def connect_and_create_resources(cls, browser_id: str, context_options: Dict[str, Any]) -> Tuple[
        Playwright, Browser, BrowserContext]:
        """连接到共享浏览器，创建资源，并增加引用计数。"""
        await cls._ensure_browser_is_running(browser_id)

        with cls._init_lock:
            cdp_endpoint = cls._context_map[browser_id]["cdp_endpoint"]

        playwright_cm = async_playwright()
        p = await playwright_cm.__aenter__()

        try:
            browser_connection = await p.chromium.connect_over_cdp(cdp_endpoint, timeout=cls._timeout)
            context = await browser_connection.new_context(**context_options)

            with cls._init_lock:
                if browser_id in cls._context_map:
                    cls._context_map[browser_id]["ref_count"] += 1
                    print(
                        f"INFO [PlaywrightManager]: Connection successful for browser ID '{browser_id}'. Ref count is now {cls._context_map[browser_id]['ref_count']}.")
                else:
                    raise RuntimeError(f"Browser for ID '{browser_id}' disappeared after launch.")

            return p, browser_connection, context
        except Exception as e:
            await playwright_cm.__aexit__(None, None, None)
            print(f"ERROR [PlaywrightManager]: Failed to connect to browser ID '{browser_id}'. Error: {e}")
            raise

    @classmethod
    async def release_connection(cls, browser_id: str):
        """减少一个连接的引用计数，并在计数为零时关闭浏览器。"""
        browser_process_handler = None
        with cls._init_lock:
            if browser_id not in cls._context_map:
                return

            cls._context_map[browser_id]["ref_count"] -= 1
            ref_count = cls._context_map[browser_id]["ref_count"]
            print(
                f"INFO [PlaywrightManager]: Releasing connection for browser ID '{browser_id}'. Ref count is now {ref_count}.")

            if ref_count <= 0:
                print(
                    f"INFO [PlaywrightManager]: Ref count for browser ID '{browser_id}' is zero. Shutting down browser.")
                browser_info = cls._context_map.pop(browser_id)
                browser_process_handler = browser_info.get("browser_process_handler")

        if browser_process_handler and browser_process_handler.is_connected():
            await browser_process_handler.close()
            print(f"INFO [PlaywrightManager]: Browser for ID '{browser_id}' has been shut down.")

    @classmethod
    async def shutdown_all(cls):
        """强制关闭所有由管理器启动的浏览器进程。"""
        with cls._init_lock:
            browser_ids = list(cls._context_map.keys())
            if not browser_ids: return

            print(f"INFO [PlaywrightManager]: Forcefully shutting down all {len(browser_ids)} browser processes.")
            infos_to_close = [cls._context_map.pop(bid) for bid in browser_ids]

        for info in infos_to_close:
            handler = info.get("browser_process_handler")
            if handler and handler.is_connected():
                await handler.close()

        print("INFO [PlaywrightManager]: All shared browser processes have been shut down.")


async def test_browser_crash_and_restart_logic():
    """
    测试用例：验证浏览器崩溃后的重启逻辑
    """
    # --- 1. 初始化和设置 ---
    # 为了测试隔离，我们为 PlaywrightManager 设置一个独立的 _context_map
    # 这样不会影响到其他可能并行运行的测试
    PlaywrightManager._context_map = {}
    manager = PlaywrightManager(headless=True)
    browser_id = "test_crash_recovery_browser"
    context_options = {"viewport": {"width": 1280, "height": 720}}

    # --- 2. 第一次连接，正常启动浏览器 ---
    print(f"\n--- [Step 1] Initial connection for browser_id '{browser_id}' ---")

    p1, browser_conn1, context1 = None, None, None
    try:
        p1, browser_conn1, context1 = await manager.connect_and_create_resources(
            browser_id=browser_id,
            context_options=context_options
        )

        # 验证初始状态
        assert browser_id in manager._context_map
        initial_info = manager._context_map[browser_id]
        initial_handler = initial_info["browser_process_handler"]
        initial_port = initial_info["port"]
        initial_ref_count = initial_info["ref_count"]

        print(f"Initial browser launched successfully on port {initial_port}.")
        print(f"Initial ref_count: {initial_ref_count}")

        assert initial_handler.is_connected()
        assert initial_ref_count == 1

        # 验证浏览器可以正常工作
        page = await context1.new_page()
        await page.goto("about:blank")
        assert await page.title() == ""
        await page.close()
        print("Initial browser is responsive.")

        # --- 3. 模拟浏览器崩溃 ---
        print(f"\n--- [Step 2] Simulating browser crash by closing process on port {initial_port} ---")
        await initial_handler.close()

        # 验证浏览器确实已关闭
        assert not initial_handler.is_connected()
        # _is_browser_responsive 应该返回 False
        assert not manager._is_browser_responsive(initial_port)
        print("Browser crash simulated successfully.")

        # --- 4. 再次连接，触发恢复逻辑 ---
        print(f"\n--- [Step 3] Re-connecting with same browser_id to trigger recovery ---")

        # 为了更好地观察日志，可以 patch print 函数
        with patch('builtins.print') as mocked_print:
            p2, browser_conn2, context2 = await manager.connect_and_create_resources(
                browser_id=browser_id,
                context_options=context_options
            )
            # 检查是否有重启日志
            restart_warning_found = any(
                f"unresponsive. Relaunching." in str(call) for call in mocked_print.call_args_list
            )
            assert restart_warning_found, "The 'Relaunching' warning message was not printed."
            print("\n[Mocked Print Output Indicates Relaunch]")

        # --- 5. 验证恢复后的状态 ---
        print("\n--- [Step 4] Verifying state after recovery ---")
        assert browser_id in manager._context_map

        restarted_info = manager._context_map[browser_id]
        restarted_handler = restarted_info["browser_process_handler"]
        restarted_port = restarted_info["port"]
        restarted_ref_count = restarted_info["ref_count"]

        print(f"Browser relaunched on new port {restarted_port}.")
        print(f"Ref_count after relaunch: {restarted_ref_count}")

        # 断言这是一个新的浏览器实例
        assert restarted_handler is not initial_handler, "A new browser handler should have been created."
        # 新端口很可能与旧端口不同（但不是绝对保证，取决于操作系统端口释放速度）
        # 所以检查句柄是更可靠的断言
        assert restarted_handler.is_connected(), "Relaunched browser should be connected."

        # 引用计数在旧实例被清理后，新实例的计数应该从1开始
        # 注意：这里的 ref_count 是针对新浏览器实例的，不是累加的
        assert restarted_ref_count == 1, "Ref count for the new browser instance should be 1."

        # 验证新浏览器可以正常工作
        restarted_page = await context2.new_page()
        await restarted_page.goto("http://example.com")
        assert "Example Domain" in await restarted_page.title()
        await restarted_page.close()
        print("Relaunched browser is responsive and working correctly.")

    finally:
        # --- 6. 清理 ---
        print("\n--- [Step 5] Cleaning up resources ---")
        if browser_id in manager._context_map:
            await manager.release_connection(browser_id)
            print(f"Released connection for browser_id '{browser_id}'.")

        # 验证清理后，map 中不再有此 browser_id
        assert browser_id not in manager._context_map
        print("Cleanup complete. Test finished.")

