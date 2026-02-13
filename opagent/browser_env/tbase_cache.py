import requests
import json
import logging
import os
import concurrent.futures
import time
from threading import Lock

# --- 日志设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class CookieCache:
    # --- 静态配置 ---
    BASE_URL = "https://megadata.antgroup-inc.cn/api/v1/data/cache"
    COOKIE = "spanner=6iTma+SBALeMnXhpHEN2yWxROgo9NkOZ"
    REMOTE_EXPIRE_TIME = 60 * 60 * 24 * 14  # 远程缓存14天

    # --- 本地缓存和重试配置 ---
    # 注释修正：60 * 60 是 1 小时
    LOCAL_CACHE_TTL = 60 * 60  # 本地缓存过期时间, 3600秒 = 1小时
    MAX_RETRIES = 5        # 最大重试次数
    RETRY_DELAY = 1        # 重试间隔时间, 1秒

    def __init__(self, perfix="cookie_1d_"):
        """
        初始化CookieCache实例，包含一个用于本地缓存的字典和线程锁。
        """
        self.local_cache = {}
        self.lock = Lock()
        self.perfix = perfix

    def _invalidate_local_cache(self, domain: str):
        """
        使指定域名的本地缓存失效。
        """
        with self.lock:
            if domain in self.local_cache:
                del self.local_cache[domain]
                logger.info(f"本地缓存已失效, domain: {domain}")
    
    def _fetch_from_remote(self, domain: str) -> list | None:
        """
        【新】从远程API获取并处理cookie数据。
        - 成功则返回处理后的cookies列表。
        - 遇到不可重试的错误(如404, JSON解析错误)则返回None。
        - 遇到可重试的错误(如Timeout, ConnectionError)则直接抛出异常。
        """
        key = f"{self.perfix}{domain.replace('.', '_')}"
        url = f"{self.BASE_URL}/{key}"
        headers = {"Cookie": self.COOKIE}
        
        # 删除代理环境变量
        if 'HTTP_PROXY' in os.environ: del os.environ['HTTP_PROXY']
        if 'HTTPS_PROXY' in os.environ: del os.environ['HTTPS_PROXY']

        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            data_wrapper = response.json()
            data = data_wrapper.get("data")

            if not data:
                #logger.warning(f"查询的cookie不存在或返回数据为空, domain: {domain}")
                return None  # 不可重试错误

            cookies_data = json.loads(data)

            if "cookie" in self.perfix:
                cookies = cookies_data['cookies'] if isinstance(cookies_data, dict) and 'cookies' in cookies_data else cookies_data

                if not isinstance(cookies, list):
                    logger.warning(f"期望得到cookie列表但类型为 {type(cookies)}, domain: {domain}")
                    return None  # 不可重试错误

                for cookie in cookies:
                    if "sameSite" not in cookie or cookie["sameSite"] not in ("Lax", "Strict", "None"):
                        cookie["sameSite"] = "Lax"
                
                return cookies # 成功
            else:
                return cookies_data

        except (requests.exceptions.HTTPError, json.JSONDecodeError) as e:
            # 这些是明确的、不可重试的错误
            logger.error(f"读取或解析数据时发生不可重试错误, domain:{domain}, error: {e}")
            return None
        # 注意：Timeout和ConnectionError等可重试异常会在此处被直接抛出，由上层调用者处理

    def get_session(self, domain: str) -> list:
        """
        【重构】获取指定域名的session (cookies)，采用"缓存-重试-再缓存检查"策略。
        1. 检查一级本地缓存 (In-Memory)。
        2. 若未命中，则进入重试循环，尝试从远程API获取。
           - 在每次重试网络请求前，都会再次检查本地缓存，以防其他线程已填充。
        """
        # --- 1. 检查一级本地缓存 (快速路径) ---
        with self.lock:
            cached_entry = self.local_cache.get(domain)
        if cached_entry:
            cached_data, timestamp = cached_entry
            if time.time() - timestamp < self.LOCAL_CACHE_TTL:
                logger.info(f"命中本地缓存 (L1), domain: {domain}")
                return cached_data
            else:
                logger.info(f"本地缓存 (L1) 已过期, domain: {domain}")
                self._invalidate_local_cache(domain)

        logger.info(f"本地缓存未命中，开始从远程获取 (带重试), domain: {domain}")
        cookies = []
        # --- 2. 远程获取与重试循环 ---
        for attempt in range(self.MAX_RETRIES):
            try:
                # 尝试从远程获取数据
                cookies = self._fetch_from_remote(domain)

                if cookies is not None:
                    # 成功获取
                    logger.info(f"成功从远程获取cookie, domain: {domain}")
                    with self.lock:
                        self.local_cache[domain] = (cookies, time.time())
                    return cookies
                else:
                    # _fetch_from_remote返回None，表示不可重试的错误
                    return []

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                logger.warning(f"读取数据失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}), domain:{domain}, error: {e}")
                
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                    
                    # 【关键改进】在下一次重试前，再次检查缓存
                    with self.lock:
                        cached_entry = self.local_cache.get(domain)
                    if cached_entry:
                        logger.info(f"在重试间隔期间，从缓存中获取到了数据, domain: {domain}")
                        # 假设数据不过期，因为是刚刚被别的线程放入的
                        return cached_entry[0]
                else:
                    logger.error(f"所有重试均失败，无法获取cookie, domain: {domain}")

        return [] # 所有重试都失败后，返回None


# 创建单例对象
cookie_cache = CookieCache(perfix="cookie_1d_") #cookie_1d_
origins_cache = CookieCache(perfix="origins_")

if __name__ == '__main__':
    # --- 并发测试配置 ---
    target_domains = ["dianping.com"]
    num_requests = 1
    max_workers = 1
    time.sleep(1)

    logger.info(f"\n开始对'{target_domains}'进行 {num_requests} 次并发请求，使用 {max_workers} 个工作线程...")

    def fetch_cookie_task(domain_to_fetch):
        """单个线程执行的任务。成功返回True，失败返回False。"""
        result = cookie_cache.get_session(domain_to_fetch)
        storage_state = origins_cache.get_session(domain_to_fetch)
        print(f"domain_to_fetch: {domain_to_fetch}, ---- \n cookie: {result},---- \n origins: {storage_state}")
        return (domain_to_fetch, result is not None and storage_state is not None)

    start_time = time.time()
    success_count = 0
    failure_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 轮流请求不同的domain
        futures = [executor.submit(fetch_cookie_task, target_domains[i % len(target_domains)]) for i in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                domain, success = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                    logger.warning(f"一次对 {domain} 的请求失败")
            except Exception as exc:
                logger.error(f"一个请求产生了异常: {exc}")
                failure_count += 1

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "="*40)
    print("      并发请求测试摘要")
    print("="*40)
    print(f"总请求次数: {num_requests}")
    print(f"成功请求:   {success_count}")
    print(f"失败请求:   {failure_count}")
    print(f"总耗时:     {total_time:.2f} 秒")
    if total_time > 0:
        print(f"每秒请求数 (RPS): {num_requests / total_time:.2f}")
    print("="*40 + "\n")

    logger.info("最后对每个domain获取一次以显示cookie数据...")
    for domain in target_domains:
        final_result = cookie_cache.get_session(domain)
        if final_result:
            print(f"--- 成功获取 {domain} 的cookie数据 (可能来自缓存) ---")
            # print(json.dumps(final_result, indent=2, ensure_ascii=False))
        else:
            print(f"--- 最后一次获取 {domain} 的cookie数据失败 ---")

