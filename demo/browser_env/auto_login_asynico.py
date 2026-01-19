"""
Script to automatically login each website using asynchronous Playwright.
(Revised version with concurrency control to prevent timeouts)
"""
import argparse
import glob
import os
import asyncio
from itertools import combinations
from pathlib import Path
from typing import List, Dict

# Make sure to install playwright and its browser dependencies
# pip install playwright
# playwright install
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import concurrent.futures
from .utils import with_timeout_legacy, change_mainip2ecsip
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
WEBARENA_AUTH_PATH = os.environ["WEBARENA_AUTH_PATH"]
VLM_EXP_DEBUG = os.environ.get('VLM_EXP_DEBUG', '0')
# --- Configuration and Constants ---

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "classifieds": {"username": "blake.sullivan@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
}

REDDIT = os.environ.get("REDDIT", "")
SHOPPING = os.environ.get("SHOPPING", "")
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
GITLAB = os.environ.get("GITLAB", "")
CLASSIFIEDS = os.environ.get("CLASSIFIEDS", "")
MAP = os.environ.get("MAP", "")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "")

DATASET = os.environ.get("DATASET", "webarena")
if DATASET == "webarena":
    SITES = ["gitlab", "shopping", "shopping_admin", "reddit", "map", "wikipedia"]
    URLS = [
        f"{GITLAB}/-/profile",
        f"{SHOPPING}/wishlist/",
        f"{SHOPPING_ADMIN}/admin/dashboard",
        f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
        f"{MAP}",
        f"{WIKIPEDIA}",
    ]
    EXACT_MATCH = [True, True, True, True, True, True]
    KEYWORDS = ["", "", "Dashboard", "Delete", "OpenStreetMap","Wikipedia"]
elif DATASET == "visualwebarena":
    SITES = ["shopping", "reddit", "classifieds"]
    URLS = [
        f"{SHOPPING}/wishlist/",
        f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
        f"{CLASSIFIEDS}/index.php?page=user&action=items",
    ]
    EXACT_MATCH = [True, True, True]
    KEYWORDS = ["", "Delete", "My listings"]
else:
    raise ValueError(f"Dataset not implemented: {DATASET}")
print(f"URLS: {URLS}")
# --- Concurrency and Timeout Settings ---
# Adjust this value based on your machine's CPU/RAM. 3-5 is a safe start.
MAX_CONCURRENT_BROWSERS = 5
# Default timeout for Playwright actions (in milliseconds)
DEFAULT_TIMEOUT = 20 * 60 * 1000 # 60 seconds

HEADLESS = True
SLOW_MO = 300  # Keep a slight delay for stability
WEBARENA_PROXY = os.environ.get("WEBARENA_PROXY", None)
#WEBARENA_PROXY = None #
print(f"auto login WEBARENA_PROXY {WEBARENA_PROXY}")

BROWSER_ARGS = [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu', # Often helps in headless environments
]
EXECUTABLE_PATH = "/root/.cache/ms-playwright/chromium-1055/chrome-linux/chrome-wrapper"

assert len(SITES) == len(URLS) == len(EXACT_MATCH) == len(KEYWORDS)


# --- Asynchronous Functions ---

async def async_is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True, owner_actor=None
) -> bool:
    """Asynchronously test whether the cookie is expired."""
    if not storage_state.exists():
        return True

    browser = owner_actor.browser_unit.browser

    context = await browser.new_context(storage_state=storage_state,
    ignore_https_errors=True,
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    )
    context.set_default_timeout(DEFAULT_TIMEOUT)
    page = await context.new_page()
    try:
        await page.goto(url)
        await page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT)
        d_url = page.url
        content = await page.content()

        logger.info(f"Verifying URL: {d_url} (Expected similar to: {url})")

        if keyword:
            return keyword not in content

        return (d_url != url) if url_exact else (url not in d_url)
    except PlaywrightTimeoutError as e:
        logger.error(f"Timeout error during verification for {url}: {e}")
        return True # Assume expired if a timeout occurs
    except Exception as e:
        logger.error(f"An unexpected error occurred during verification for {url}: {e}")
        return True # Assume expired if any other error occurs
    finally:
        await context.close()
        logger.info(f"Closed finished browser for {url}")



async def async_renew_comb(comb: List[str], auth_folder: str = "./.auth", REPLACE_WITH_YOUR_HOST = None, owner_actor=None) -> None:
    """
    Asynchronously renew cookies for a combination of sites, with a retry mechanism.
    """
    max_retries = 10
    base_delay = 5  # 基础延迟时间（秒），用于指数退避

    if VLM_EXP_DEBUG == '1':
        max_retries = 3
        base_delay = 1

    if REPLACE_WITH_YOUR_HOST is not None:
        shopping_url = change_mainip2ecsip(SHOPPING, REPLACE_WITH_YOUR_HOST)
        reddit_url = change_mainip2ecsip(REDDIT, REPLACE_WITH_YOUR_HOST)
        classifieds_url = change_mainip2ecsip(CLASSIFIEDS, REPLACE_WITH_YOUR_HOST)
        shopping_admin_url = change_mainip2ecsip(SHOPPING_ADMIN, REPLACE_WITH_YOUR_HOST)
        gitlab_url = change_mainip2ecsip(GITLAB, REPLACE_WITH_YOUR_HOST)

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for combination: {comb} REPLACE_WITH_YOUR_HOST: {REPLACE_WITH_YOUR_HOST}")
            
            browser = owner_actor.browser_unit.browser
            context = None
            try:
                context = await browser.new_context(
                ignore_https_errors=True,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
                )
                context.set_default_timeout(DEFAULT_TIMEOUT)
                page = await context.new_page()

                if "shopping" in comb:
                    username, password = ACCOUNTS["shopping"]["username"], ACCOUNTS["shopping"]["password"]
                    await page.goto(f"{shopping_url}/customer/account/login/", timeout=DEFAULT_TIMEOUT)
                    await page.get_by_label("Email", exact=True).fill(username)
                    await page.get_by_label("Password", exact=True).fill(password)
                    await page.get_by_role("button", name="Sign In").click()
                    await page.wait_for_url(f"{shopping_url}/customer/account/", timeout=DEFAULT_TIMEOUT)

                if "reddit" in comb:
                    username, password = ACCOUNTS["reddit"]["username"], ACCOUNTS["reddit"]["password"]
                    await page.goto(f"{reddit_url}/login", timeout=DEFAULT_TIMEOUT)
                    await page.get_by_label("Username").fill(username)
                    await page.get_by_label("Password").fill(password)
                    await page.get_by_role("button", name="Log in").click()

                if "classifieds" in comb:
                    username, password = ACCOUNTS["classifieds"]["username"], ACCOUNTS["classifieds"]["password"]
                    await page.goto(f"{classifieds_url}/index.php?page=login", timeout=DEFAULT_TIMEOUT)
                    await page.locator("#email").fill(username)
                    await page.locator("#password").fill(password)
                    await page.get_by_role("button", name="Log in").click()

                if "shopping_admin" in comb:
                    username, password = ACCOUNTS["shopping_admin"]["username"], ACCOUNTS["shopping_admin"]["password"]
                    await page.goto(f"{shopping_admin_url}/login", timeout=DEFAULT_TIMEOUT)
                    await page.get_by_placeholder("user name").fill(username)
                    await page.get_by_placeholder("password").fill(password)
                    await page.get_by_role("button", name="Sign in").click()

                if "gitlab" in comb:
                    username, password = ACCOUNTS["gitlab"]["username"], ACCOUNTS["gitlab"]["password"]
                    await page.goto(f"{gitlab_url}/users/sign_in", timeout=DEFAULT_TIMEOUT)
                    await page.get_by_test_id("username-field").fill(username)
                    await page.get_by_test_id("password-field").fill(password)
                    await page.get_by_test_id("sign-in-button").click()

                # Wait for the last login to complete and the page to settle
                await page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT)

                storage_path = Path(auth_folder) / f"{'.'.join(sorted(comb))}_state.json"
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                await context.storage_state(path=storage_path)
                
                logger.info(f"✅ Successfully saved state for {comb} to {storage_path} on attempt {attempt + 1}.")
                return  # 成功，退出函数

            finally:
                await context.close()
                #await browser.close()
                logger.info(f"Closed finished browser for {comb}")

        except PlaywrightTimeoutError as e:
            error_message = e.message.splitlines()[0]
            logger.error(f"❌ Attempt {attempt + 1} failed for {comb} with a timeout: {error_message}")
        except Exception as e:
            logger.error(f"❌ Attempt {attempt + 1} failed for {comb} with an unexpected error: {e}")

        # 如果不是最后一次尝试，则等待后重试
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)  # 指数退避
            logger.info(f"   Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
        else:
            # 所有重试均失败
            logger.error(f"❌ All {max_retries} attempts failed for {comb}. Giving up.")


def get_site_comb_from_filepath(file_path: str) -> List[str]:
    """Helper function to extract site combination from filename."""
    return os.path.basename(file_path).rsplit("_", 1)[0].split(".")

# --- Worker Functions with Semaphore Control ---

async def renew_comb_worker(comb: List[str], auth_folder: str, REPLACE_WITH_YOUR_HOST, owner_actor=None):
    """A worker that acquires the semaphore before running the renewal task."""
    await async_renew_comb(comb, auth_folder, REPLACE_WITH_YOUR_HOST, owner_actor)

async def verify_cookie_worker(verification_info: Dict, owner_actor=None):
    """A worker that acquires the semaphore before running the verification task."""
    #async with semaphore:
    c_file = verification_info["file"]
    cur_site = verification_info["site"]
    url = verification_info["url"]
    keyword = verification_info["keyword"]
    exact_match = verification_info["exact_match"]
    
    is_expired = await async_is_expired(Path(c_file), url, keyword, exact_match, owner_actor=owner_actor)
    return {"is_expired": is_expired, "meta": verification_info}

async def async_verif_web(auth_folder: str, REPLACE_WITH_YOUR_HOST = None, url_list = URLS, owner_actor=None) -> None:
    """
    Concurrently verify all generated cookie files using pure asyncio.
    """
    logger.info("\n--- Starting Concurrent Cookie Verification ---")
    cookie_files = list(glob.glob(f"{auth_folder}/*.json"))
    if not cookie_files:
        logger.error("No cookie files found to verify.")
        return

    if REPLACE_WITH_YOUR_HOST is not None:
        url_list = [change_mainip2ecsip(url, REPLACE_WITH_YOUR_HOST) for url in url_list]

    # 1. 准备所有任务的元数据
    verification_tasks_meta = []
    for c_file in cookie_files:
        comb = get_site_comb_from_filepath(c_file)
        for cur_site in comb:
            try:
                site_idx = SITES.index(cur_site)
                verification_meta = {
                    "file": c_file, 
                    "site": cur_site, 
                    "url": url_list[site_idx],
                    "keyword": KEYWORDS[site_idx],
                    "exact_match": EXACT_MATCH[site_idx]
                }
                verification_tasks_meta.append(verification_meta)
            except ValueError:
                logger.warning(f"Warning: Site '{cur_site}' from file '{c_file}' not in configured SITES list. Skipping.")

    if not verification_tasks_meta:
        logger.info("No valid verification tasks to run.")
        return
        
    # 2. 使用 asyncio.create_task 启动所有并发任务
    # 同样，不再使用 owner_actor.submit()
    tasks = [
        asyncio.create_task(
            verify_cookie_worker(meta, owner_actor=owner_actor)
        ) 
        for meta in verification_tasks_meta
    ]

    logger.info(f"Starting {len(tasks)} cookie verification workers.")

    # 3. 使用 asyncio.gather 等待所有任务完成
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 4. 统一处理所有结果
    logger.info("\n--- Verification Results ---")
    for i, res in enumerate(results):
        meta = verification_tasks_meta[i] # 获取对应的元数据
        if isinstance(res, Exception):
            logger.error(f"❌ Error verifying site '{meta['site']}' from file '{os.path.basename(meta['file'])}'. Error: {res}")
        else:
            # res 就是 verify_cookie_worker 返回的字典
            if res['is_expired']:
                logger.error(f"🔴 [EXPIRED] Cookie for site '{res['meta']['site']}' in file '{os.path.basename(res['meta']['file'])}' is expired. REPLACE_WITH_YOUR_HOST {REPLACE_WITH_YOUR_HOST}")
            else:
                logger.info(f"✅ [VALID] Cookie for site '{res['meta']['site']}' in file '{os.path.basename(res['meta']['file'])}' is valid. REPLACE_WITH_YOUR_HOST {REPLACE_WITH_YOUR_HOST}")

@with_timeout_legacy(60 * 60)
async def async_generate_new_cookies(auth_folder: str, REPLACE_WITH_YOUR_HOST = None, owner_actor=None) -> None:
    """Concurrently generate new cookies for all site combinations with controlled concurrency."""
    print(f"--- Starting Concurrent Cookie Generation (max {MAX_CONCURRENT_BROWSERS} at a time) ---")
    os.makedirs(auth_folder, exist_ok=True)
    
    combs_to_generate = []
    pairs = list(combinations(SITES, 2))
    for pair in pairs:
        if "reddit" in pair and ("shopping" in pair or "shopping_admin" in pair):
            continue
        if "map" in pair or "wikipedia" in pair:
            continue
        combs_to_generate.append(list(sorted(pair)))
    
    for site in SITES:
        combs_to_generate.append([site])


    tasks = []
    for comb in combs_to_generate:
        # 你在这里，可以直接创建任务，而不需要 submit
        # 因为你已经在目标事件循环里了！
        task = asyncio.create_task(
            renew_comb_worker(comb, auth_folder, REPLACE_WITH_YOUR_HOST, owner_actor=owner_actor)
        )
        tasks.append(task)
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"name Error in generating cookies for comb {combs_to_generate[i]}: {res}")

    logger.info("\n--- Cookie Generation Complete ---")
    await async_verif_web(auth_folder, REPLACE_WITH_YOUR_HOST, owner_actor=owner_actor)
    


async def main():
    """Main execution function to parse arguments and run async tasks."""
    parser = argparse.ArgumentParser(description="Asynchronously generate and verify web cookies.")
    parser.add_argument("--site_list", nargs="+", default=[], help="A specific list of sites to renew cookies for.")
    parser.add_argument("--auth_folder", type=str, default=WEBARENA_AUTH_PATH+'/.auth/', help="Folder to store authentication files.")
    parser.add_argument("--verify_only", action="store_true", help="Only run the verification process.")
    args = parser.parse_args()

    if args.verify_only:
        await async_verif_web(args.auth_folder)
    elif not args.site_list:
        await async_generate_new_cookies(auth_folder=args.auth_folder)
    else:
        # Renew a single specified combination (doesn't need semaphore as it's only one task)
        print(f"Renewing single combination: {sorted(args.site_list)}")
        await async_renew_comb(sorted(args.site_list), auth_folder=args.auth_folder)
        await async_verif_web(args.auth_folder)


if __name__ == "__main__":
    if not all([REDDIT, SHOPPING, SHOPPING_ADMIN, GITLAB]):
        print("Warning: Not all environment variables for site URLs are set. This may cause errors.")
    
    asyncio.run(main())
