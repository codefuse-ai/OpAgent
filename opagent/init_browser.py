import asyncio
import json
import os
import socket
import traceback
from typing import Optional, List, Dict
import requests
import time
from playwright.async_api import async_playwright, Browser, Playwright
import signal
import logging
import sys

HOST_PORTS = os.environ.get("HOST_PORTS", "")
preferred_ports_list = [int(port) for port in HOST_PORTS.split(',')] if HOST_PORTS else []
NUM_BROWSERS = int(os.environ.get("NUM_BROWSERS", "20"))
TENSORBOARD_DIR = os.environ.get("TENSORBOARD_DIR")
TASK_ID = os.environ.get("TASK_ID")
BROWSER_OUTPUT_PATH = TENSORBOARD_DIR.replace("tensorboard", "browser_config")
BROWSER_OUTPUT_PATH = BROWSER_OUTPUT_PATH + "/" + TASK_ID + "/"
print("BROWSER_OUTPUT_PATH: ", BROWSER_OUTPUT_PATH)
try:
    os.rmdir(BROWSER_OUTPUT_PATH)
    print(f"成功删除空目录: {BROWSER_OUTPUT_PATH}")
except OSError as e:
    # 常见的错误包括：目录不存在(FileNotFoundError), 目录不为空(OSError: [Errno 39] Directory not empty)
    print(f"删除目录 {BROWSER_OUTPUT_PATH} 时出错: {e}")

os.makedirs(BROWSER_OUTPUT_PATH, exist_ok=True)

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
output_log = f"{BROWSER_OUTPUT_PATH}/{hostname}_{ip_address}_init_browser.log"

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[
        logging.FileHandler(output_log),  # 输出到文件
        logging.StreamHandler(sys.stdout)       # 同时输出到控制台
    ]
)

logging.info("配置浏览器日志")

class BrowserLauncher:
    """负责在当前机器上启动、持久化并监控浏览器实例，在异常退出时自动重启。"""
    def __init__(self, headless: bool = True, slow_mo: int = 0, BROWSER_OUTPUT_PATH: str = "."):
        self.headless = headless
        self.slow_mo = slow_mo
        self.BROWSER_OUTPUT_PATH = BROWSER_OUTPUT_PATH
        self.executable_path = "/root/.cache/ms-playwright/chromium-1055/chrome-linux/chrome-wrapper"
        
        self.playwright: Optional[Playwright] = None
        self._managed_browsers: Dict[str, Browser] = {}
        self.lock = asyncio.Lock()
        self.port_lock = asyncio.Lock()
        self.shutdown_event = asyncio.Event()

    def _find_port_from_list(self, ports: List[int]) -> Optional[int]:
        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                    return port
            except OSError:
                continue
        return None

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _get_available_port(self, preferred_ports: Optional[List[int]] = None) -> int:
        if preferred_ports:
            port = self._find_port_from_list(preferred_ports)
            if port is not None:
                logging.info(f"从指定列表中找到可用端口: {port}")
                return port
        port = self._find_free_port()
        logging.info(f"由系统分配可用端口: {port}")
        return port

    def get_websocket_url(self, port: int) -> Optional[str]:
        for _ in range(5):
            try:
                response = requests.get(f"http://localhost:{port}/json/version/", proxies=None, timeout=5)
                response.raise_for_status()
                return response.json()["webSocketDebuggerUrl"]
            except Exception as e:
                logging.info(f"等待浏览器就绪... 无法在端口 {port} 获取WebSocket URL: {e}")
                time.sleep(1)
        logging.error(f"错误：多次尝试后仍无法获取WebSocket调试URL:{traceback.print_exc()}")
        
        return None
    
    async def launch_single_browser_with_retries(self, browser_id):
        if not self.playwright:
            raise RuntimeError("Playwright 实例尚未初始化。")

        for attempt in range(5):
            port = -1
            browser = None
            
            # 使用锁来安全地获取可用端口
            async with self.port_lock:
                port = self._get_available_port(preferred_ports_list)
                logging.info(f"浏览器 {browser_id}: 已锁定并尝试在端口 {port} 启动 (第 {attempt + 1} 次尝试)...")
                
                launch_options = {
                    "headless": self.headless, "slow_mo": self.slow_mo,
                    "args": [
                        '--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage',
                        '--disable-gpu', '--no-zygote', f'--remote-debugging-port={port}',
                        '--remote-debugging-address=0.0.0.0',
                    ]
                }
                if self.executable_path and os.path.exists(self.executable_path):
                    launch_options["executable_path"] = self.executable_path
                
                try:
                    browser = await self.playwright.chromium.launch(**launch_options)
                except Exception as e:
                    logging.error(f"错误: 浏览器 {browser_id}: 在端口 {port} 上启动失败。错误: {e}")
                    if browser: await browser.close()
                    await asyncio.sleep(1)
                    continue
            
            if browser:
                logging.info(f"浏览器 {browser_id}: 在端口 {port} 上启动成功！")
                cdp_endpoint_ws = self.get_websocket_url(port)
                if not cdp_endpoint_ws:
                    logging.warning(f"警告: 浏览器 {browser_id} 在端口 {port} 启动，但无法获取 WebSocket URL。")
                    await browser.close()
                    continue
                
                hostname = socket.gethostname()
                ip_address = socket.gethostbyname(hostname)
                cdp_endpoint_ip = cdp_endpoint_ws.replace("127.0.0.1", ip_address).replace("localhost", ip_address)
                return browser, cdp_endpoint_ip

        logging.error(f"浏览器 {browser_id}: 达到最大重试次数后仍无法启动。")
        return None, None

    async def _update_endpoints_file(self):
        """将当前所有受管理的浏览器的Endpoints写入JSON文件。"""
        async with self.lock:
            os.makedirs(self.BROWSER_OUTPUT_PATH, exist_ok=True)
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            output_file = f"{self.BROWSER_OUTPUT_PATH}/{hostname}_{ip_address}_browser_endpoints.json"
            
            active_endpoints = list(self._managed_browsers.keys())
            
            with open(output_file, 'w') as f:
                json.dump(active_endpoints, f, indent=2)
            logging.info(f"配置文件已更新: {len(active_endpoints)} 个浏览器信息写入到 {output_file}")
    
    async def _monitor_and_restart_browser(self, browser: Browser, cdp_endpoint: str):
        """
        监控单个浏览器实例。
        仅当浏览器异常断开连接时（而不是在计划关闭期间），才触发重启流程。
        """
        logging.info(f"[*] 监控已启动: {cdp_endpoint}")
        disconnected_event = asyncio.Event()

        def on_disconnect_handler():
            disconnected_event.set()
        
        browser.on("disconnected", on_disconnect_handler)

        try:
            await disconnected_event.wait()
            
            # --- 核心决策逻辑 ---
            # 检查断开连接的原因。如果是程序正在关闭，则不进行任何操作。
            if self.shutdown_event.is_set():
                logging.info(f"[*] 浏览器 {cdp_endpoint} 在计划关闭期间断开连接。")
                return # 正常退出监控

            # 如果程序没有关闭，说明是异常崩溃
            logging.warning(f"[!] 警告: 浏览器异常断开连接! Endpoint: {cdp_endpoint}")
            await self._handle_browser_disconnection(cdp_endpoint)

        except Exception as e:
            # 仅在非计划关闭时，处理监控任务本身的异常
            if not self.shutdown_event.is_set():
                logging.error(f"[!] 错误: 监控任务 '{cdp_endpoint}' 异常退出: {e}")
                await self._handle_browser_disconnection(cdp_endpoint)
        finally:
            browser.remove_listener("disconnected", on_disconnect_handler)

    async def _handle_browser_disconnection(self, old_cdp_endpoint: str):
        """处理浏览器异常断开连接后的重启逻辑。"""
        # 双重保险：如果此时程序已经决定要关闭，则不进行重启
        if self.shutdown_event.is_set():
            logging.info(f"[*] 取消重启 {old_cdp_endpoint}，因为程序正在关闭。")
            return

        async with self.lock:
            if old_cdp_endpoint not in self._managed_browsers:
                logging.info(f"[*] 浏览器 {old_cdp_endpoint} 已被其他任务处理，跳过重启。")
                return
            
            logging.info(f"[*] 正在尝试重启浏览器以替换: {old_cdp_endpoint}")
            self._managed_browsers.pop(old_cdp_endpoint, None)
            
            # 使用更清晰的ID进行重启
            restarted_id = f"restarted_for_{old_cdp_endpoint.split(':', 2)[-1].split('/')[0]}"
            new_browser, new_cdp_endpoint = await self.launch_single_browser_with_retries(restarted_id)
            
            if new_browser and new_cdp_endpoint:
                logging.info(f"[+] 新浏览器已成功启动: {new_cdp_endpoint}")
                self._managed_browsers[new_cdp_endpoint] = new_browser
                asyncio.create_task(self._monitor_and_restart_browser(new_browser, new_cdp_endpoint))
            else:
                logging.error(f"[!] 错误: 无法重启替换 {old_cdp_endpoint} 的浏览器。当前可用浏览器数量减少。")
            
        await self._update_endpoints_file()

    async def _launch_and_monitor_initial_browsers(self, num_browsers: int):
        """启动 n 个浏览器实例，并为每个实例启动监控。"""
        logging.info(f"开始启动 {num_browsers} 个浏览器...")
        tasks = [self.launch_single_browser_with_retries(i + 1) for i in range(num_browsers)]
        results = await asyncio.gather(*tasks)

        # 使用锁来安全地修改共享字典
        async with self.lock:
            for browser, cdp_endpoint_ip in results:
                if browser and cdp_endpoint_ip:
                    self._managed_browsers[cdp_endpoint_ip] = browser
                    asyncio.create_task(self._monitor_and_restart_browser(browser, cdp_endpoint_ip))

        logging.info(f"初始启动完成: {len(self._managed_browsers)} / {num_browsers} 个浏览器成功启动并开始监控。")

        if not self._managed_browsers:
            logging.warning("警告: 未能成功启动任何浏览器。")
        else:
            await self._update_endpoints_file()

    def _handle_shutdown_signal(self):
        logging.info("收到关闭信号(SIGINT/SIGTERM), 准备执行关闭流程...")
        self.shutdown_event.set()

    async def _cleanup(self):
        """关闭所有浏览器进程和 Playwright 实例。"""
        logging.info("开始清理资源...")
        async with self.lock:
            if not self._managed_browsers:
                logging.info("没有正在运行的浏览器需要关闭。")
                return
            
            logging.info(f"正在关闭 {len(self._managed_browsers)} 个浏览器...")
            close_tasks = [browser.close() for browser in self._managed_browsers.values()]
            await asyncio.gather(*close_tasks, return_exceptions=True)
            
            logging.info(f"所有受管理的浏览器均已发送关闭命令。")
            self._managed_browsers.clear()
        
        if self.playwright:
            await self.playwright.stop()
        
        logging.info("Playwright 实例已关闭。")

    async def run(self, num_browsers: int):
        """启动并管理浏览器生命周期的主方法。"""
        async with async_playwright() as p:
            self.playwright = p

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._handle_shutdown_signal)

            await self._launch_and_monitor_initial_browsers(num_browsers)

            if not self._managed_browsers:
                logging.error("错误：未能成功启动任何浏览器。程序即将退出。")
                return

            logging.info("浏览器服务已就绪，具有异常重启功能，正在持久化运行中。")

            await self.shutdown_event.wait()
            logging.info("正在执行关闭流程...")
            await self._cleanup()
            logging.info("清理完成，程序退出。")


if __name__ == "__main__":
    launcher = BrowserLauncher(BROWSER_OUTPUT_PATH=BROWSER_OUTPUT_PATH)
    try:
        asyncio.run(launcher.run(NUM_BROWSERS))
    except KeyboardInterrupt:
        logging.info("启动器被用户中断。")

