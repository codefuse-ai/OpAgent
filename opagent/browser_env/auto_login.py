"""Script to automatically login each website"""
import argparse
import glob
import os
from socket import timeout
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from playwright.sync_api import sync_playwright
from tqdm import tqdm

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "classifieds": {
        "username": "blake.sullivan@gmail.com",
        "password": "Password.123",
    },
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
}

REDDIT = os.environ.get("REDDIT", "")
SHOPPING = os.environ.get("SHOPPING", "")
print("SHOPPING: ", SHOPPING)
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
GITLAB = os.environ.get("GITLAB", "")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
MAP = os.environ.get("MAP", "")
HOMEPAGE = os.environ.get("HOMEPAGE", "")
CLASSIFIEDS = os.environ.get("CLASSIFIEDS", "")

DATASET = os.environ.get("DATASET", "webarena")

# --- 脚本常量 ---
ONE_YEAR_IN_SECONDS=31536000
ONE_YEAR_IN_MINUTES=525600

# --- 网站端口和名称映射 ---
SITE_PORT_MAP = {
    "7770": "shopping",
    "7780": "shopping_admin",
    "9999": "reddit", # Reddit/Forum
    "8023": "gitlab"
}

if DATASET == "webarena":

    SITES = ["gitlab", "shopping", "shopping_admin", "reddit"]
    URLS = [
        f"{GITLAB}/-/profile",
        f"{SHOPPING}/wishlist/",
        f"{SHOPPING_ADMIN}/dashboard",
        f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
    ]
    EXACT_MATCH = [True, True, True, True]
    KEYWORDS = ["", "", "Dashboard", "Delete"]

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

HEADLESS = True
SLOW_MO = 0
WEBARENA_PROXY = os.environ.get("WEBARENA_PROXY", None)

assert len(SITES) == len(URLS) == len(EXACT_MATCH) == len(KEYWORDS)

def is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True
) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True

    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=True, slow_mo=SLOW_MO, 
    proxy={"server": WEBARENA_PROXY} if WEBARENA_PROXY else None,
    executable_path="/root/.cache/ms-playwright/chromium-1055/chrome-linux/chrome-wrapper",
    args=[
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage'
    ])
    context = browser.new_context(storage_state=storage_state,
    ignore_https_errors=True,
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',)
    page = context.new_page()
    page.goto(url, timeout= 10 * 60 * 1000)
    time.sleep(1)
    d_url = page.url
    content = page.content()
    context_manager.__exit__()
    print(f"keyword content: {d_url} {url}")
    if keyword:
        return keyword not in content
    else:
        if url_exact:
            return d_url != url
        else:
            return url not in d_url


def renew_comb(comb: list[str], auth_folder: str = "./.auth") -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS, 
    executable_path="/root/.cache/ms-playwright/chromium-1055/chrome-linux/chrome-wrapper",
    proxy={"server": WEBARENA_PROXY} if WEBARENA_PROXY else None,
    args=[
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage'
    ])
    context = browser.new_context(
        ignore_https_errors=True,
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',)
    page = context.new_page()

    if "shopping" in comb:
        username = ACCOUNTS["shopping"]["username"]
        password = ACCOUNTS["shopping"]["password"]
        print(f"goto {SHOPPING}/customer/account/login/")
        page.goto(f"{SHOPPING}/customer/account/login/")
        page.get_by_label("Email", exact=True).fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Sign In").click()

    if "reddit" in comb:
        username = ACCOUNTS["reddit"]["username"]
        password = ACCOUNTS["reddit"]["password"]
        print(f"goto {REDDIT}/login")
        page.goto(f"{REDDIT}/login")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password").fill(password)
        page.get_by_role("button", name="Log in").click()

    if "classifieds" in comb:
        username = ACCOUNTS["classifieds"]["username"]
        password = ACCOUNTS["classifieds"]["password"]
        print(f"goto {CLASSIFIEDS}/index.php?page=login")
        page.goto(f"{CLASSIFIEDS}/index.php?page=login")
        page.locator("#email").fill(username)
        page.locator("#password").fill(password)
        page.get_by_role("button", name="Log in").click()

    if "shopping_admin" in comb:
        username = ACCOUNTS["shopping_admin"]["username"]
        password = ACCOUNTS["shopping_admin"]["password"]
        print(f"goto {SHOPPING_ADMIN}")
        page.goto(f"{SHOPPING_ADMIN}", timeout=600000)
        page.get_by_placeholder("user name").fill(username)
        page.get_by_placeholder("password").fill(password)
        page.get_by_role("button", name="Sign in").click()

    if "gitlab" in comb:
        username = ACCOUNTS["gitlab"]["username"]
        password = ACCOUNTS["gitlab"]["password"]
        print(f"goto {GITLAB}/users/sign_in")
        page.goto(f"{GITLAB}/users/sign_in")
        page.get_by_test_id("username-field").click()
        page.get_by_test_id("username-field").fill(username)
        page.get_by_test_id("username-field").press("Tab")
        page.get_by_test_id("password-field").fill(password)
        page.get_by_test_id("sign-in-button").click()

    context.storage_state(path=f"{auth_folder}/{'.'.join(comb)}_state.json")

    context_manager.__exit__()


def get_site_comb_from_filepath(file_path: str) -> list[str]:
    comb = os.path.basename(file_path).rsplit("_", 1)[0].split(".")
    return comb


def generate_new_cookies(auth_folder: str = "./.auth") -> None:
    pairs = list(combinations(SITES, 2))
    #with ThreadPoolExecutor(max_workers=8) as executor:
    for pair in pairs:
        # Auth doesn't work on this pair as they share the same cookie
        if "reddit" in pair and (
            "shopping" in pair or "shopping_admin" in pair
        ):
            continue
        # executor.submit(
        #     renew_comb, list(sorted(pair)), auth_folder=auth_folder
        # )
        renew_comb(list(sorted(pair)), auth_folder=auth_folder)
    for site in SITES:
        renew_comb([site], auth_folder=auth_folder)
        #executor.submit(renew_comb, [site], auth_folder=auth_folder)
    verif_web(auth_folder, pairs)

def verif_web(auth_folder, pairs):
    # parallel checking if the cookies are expired  
    futures = []
    cookie_files = list(glob.glob(f"{auth_folder}/*.json"))
    #with ThreadPoolExecutor(max_workers=8) as executor:
    cur_site_list = []
    for c_file in cookie_files:
        comb = get_site_comb_from_filepath(c_file)
        for cur_site in comb:
            url = URLS[SITES.index(cur_site)]
            cur_site_list.append(url)
            print(f"url: {url}")
            keyword = KEYWORDS[SITES.index(cur_site)]
            match = EXACT_MATCH[SITES.index(cur_site)]
            # future = executor.submit(
            #     is_expired, Path(c_file), url, keyword, match
            # )
            future = is_expired(Path(c_file), url, keyword, match)
            futures.append(future)
    print("len (futures) : ", len(futures), len(cur_site_list))
    expired_count = 0
    for i, future in enumerate(futures):
        if future:
            expired_count += 1
            print(f"Cookie {cur_site_list[i]} expired.")
    print(f"expired_count: {expired_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_list", nargs="+", default=[])
    parser.add_argument("--auth_folder", type=str, default="./.auth")
    args = parser.parse_args()
    
    #verif_web(args.auth_folder, pairs)
    if not args.site_list:
        generate_new_cookies(auth_folder=args.auth_folder)
    else:
        renew_comb(args.site_list, auth_folder=args.auth_folder)
