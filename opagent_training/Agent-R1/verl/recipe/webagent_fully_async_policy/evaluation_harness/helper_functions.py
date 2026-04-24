"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""
import json
from datetime import datetime, timezone
from typing import Any, Union
from urllib.parse import urlparse

import requests
from requests.exceptions import RequestException
from beartype import beartype
from beartype.typing import Dict, List
from playwright.sync_api import Page
import traceback
import time
import os
import base64
from typing import List
from PIL import Image
from io import BytesIO
from recipe.webagent_fully_async_policy.evaluation_harness.webjudge.src.qwen_vlm_request import send_chat_completion_request_shangshu
from recipe.webagent_fully_async_policy.browser_env.env_config import (
    ACCOUNTS,
    REDDIT,
    SHOPPING,
    WIKIPEDIA,
)

proxies = {
    'http': os.environ.get("HTTP_PROXY", ""),
    'https': os.environ.get("HTTPS_PROXY", ""),
}

class PseudoPage:
    def __init__(self, original_page: Page, url: str):
        self.url = url
        self.original_page = original_page

    def __getattr__(self, attr: str) -> Any:
        # Delegate attribute access to the original page object
        if attr not in ["url"]:
            return getattr(self.original_page, attr)
        else:
            return getattr(self, attr)


@beartype
def shopping_get_auth_token() -> str:
    shopping_url = SHOPPING
    response = requests.post(
        url=f"{shopping_url}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
        #proxies=proxies,
    )
    try:
        token: str = response.json()
    except Exception as e:
        print(f"error token: {response}, exception: {e}")
        token = ""
    return token


@beartype
def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""
    shopping_url = SHOPPING
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(
        f"{shopping_url}/rest/V1/orders", params=params, headers=header,
        #proxies=proxies
    )
    try:
        assert response.status_code == 200
        response_obj = response.json()["items"][0]
        order_id = int(response_obj["increment_id"])
        order_url = f"{shopping_url}/sales/order/view/order_id/{order_id}/"
        return order_url
    except Exception as e:
        print(f"shopping_get_latest_order_url error: {response}, exception: {e}")
        return ""



@beartype
def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    shopping_url = SHOPPING
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{shopping_url}/rest/V1/products/{sku}/reviews", headers=header,
        #proxies=proxies
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


@beartype
def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    shopping_url = SHOPPING
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{shopping_url}/rest/V1/products/{sku}/reviews", headers=header,
        #proxies=proxies
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


@beartype
def shopping_get_sku_latest_review_text(sku: str) -> str:
    """Get the latest review text for shopping admin."""
    shopping_url = SHOPPING
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{shopping_url}/rest/V1/products/{sku}/reviews", headers=header,
        #proxies=proxies
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    text: str = response_obj[-1]["detail"]
    return text


@beartype
def shopping_get_sku_latest_review_title(sku: str) -> str:
    """Get the latest review title for shopping admin."""
    shopping_url = SHOPPING
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{shopping_url}/rest/V1/products/{sku}/reviews", headers=header,
        #proxies=proxies
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    title: str = response_obj[-1]["title"]
    return title


@beartype
def shopping_get_sku_product_page_url(sku: str) -> str:
    """Get product page url from sku"""
    shopping_url = SHOPPING
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{shopping_url}/rest/V1/products/{sku}", headers=header,
        #proxies=proxies
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    for custom_attributes in response_obj["custom_attributes"]:
        if custom_attributes["attribute_code"] == "url_key":
            return f"{shopping_url}/{custom_attributes['value']}.html"
    return ""


async def shopping_get_all_product_order(
    page: Page | PseudoPage,
) -> List[Dict[str, str]]:
    """
    Get info of all product in a given order page.

    Example output:
    [
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Strawberry, 19oz Box (12 Bars)\nSize\n12 Count (Pack of 1)",
            "options": {
                "Size": "12 Count (Pack of 1)"
            },
            "sku": "B00MXUFL0E",
            "price": "$24.50",
            "qty": "Ordered2",
            "subtotal": "$49.00"
        },
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Chocolatey Chip Cookie Dough, 19oz Box (12 Bars)",
            "sku": "B07ZD2PB9F",
            "price": "$42.30",
            "qty": "Ordered2",
            "subtotal": "$84.60"
        }
    ]
    """
    try:
        result = await page.evaluate(
            f"""
(() => {{
    try {{
        const products = [...document.querySelector("#my-orders-table").getElementsByTagName('tbody')].map(
            (x) => {{
                return [...x.getElementsByTagName('td')].reduce(function(obj, y) {{
                    const key = y.className.split(' ')[1];
                    obj[key] = y.outerText;
                    // check if options exist
                    if (key === 'name' && y.querySelector('dl')) {{
                        var option_dict = {{}}
                        const options = [...y.querySelector('dl').children];
                        for (let i = 0; i < options.length; i += 2) {{
                            option_dict[options[i].outerText] = options[i+1].outerText;
                        }}
                        obj['options'] = option_dict;
                    }}
                    return obj;
                }}, {{}})
            }}
        );
        return products;
    }} catch (e) {{
        // If any errors are caught, return an empty string
        return e;
        return [];
    }}
}})();
            """
        )
        return result
    except Exception as e:
        result = []

    return result


async def shopping_get_order_product_name_list(page: Page | PseudoPage) -> str:
    try:
        products = await shopping_get_all_product_order(page)

        return " |OR| ".join([p["name"] for p in products])
    except Exception:
        return ""


async def shopping_get_order_product_quantity(
    page: Page | PseudoPage, sku: str
) -> int:
    try:
        if "|OR|" in sku:
            skus = sku.split(" |OR| ")
        else:
            skus = [sku]

        products = await shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() in skus:
                # Ordered{qty}
                return int(product["qty"][7:])
        return 0
    except Exception:
        return 0

async def shopping_get_order_product_option(
    page: Page | PseudoPage, sku: str, option_name: str
) -> str:
    try:
        products = await shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() == sku:
                # Ordered{qty}
                return product["options"][option_name]
        return ""
    except Exception as e:
        return ""


async def shopping_get_product_attributes(
    page: Page | PseudoPage, attribute: str
) -> str:
    # Get the values of all cells in the table for the given attribute
    try:
        result = await page.evaluate(
            f"""
                (() => {{
                try {{
                    // Create an array of search terms, splitting the string by ' |OR| '
                    const searchTerms = '{attribute}'.toLowerCase().split(' |or| ');
                    // Convert the children of the tbody inside the element with the given ID into an array
                    return Array.from(
                    document.querySelector('#productDetails_detailBullets_sections1 > tbody').children
                    )
                    // Filter the array to only include elements where the first child's text includes any of the search terms
                    .filter(x =>
                    searchTerms.some(term => x.children[0].outerText.toLowerCase().includes(term))
                    )
                    // Map over the filtered elements to get the outerText of their second child
                    .map(x => x.children[1].outerText)
                    // Join all the resulting strings with a comma and a space
                    .join(', ')
                }} catch (e) {{
                    // If any errors are caught, return an empty string
                    return ''
                }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


async def shopping_get_product_price(page: Page | PseudoPage) -> Union[float, int]:
    """Get the price of the product on the shopping website."""
    try:
        result = await page.evaluate(
            """
                (() => {{
                    res = parseFloat(document.querySelector(\"#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span\")
                    .outerText.substr(1));
                    return res ? res : 0;
                }})();
            """
        )
    except Exception:
        result = 0

    return result


async def shopping_get_num_reviews(page: Page | PseudoPage) -> int:
    """Get the price of the product on the shopping website."""
    try:
        result = await page.evaluate(
            """
                (() => {{
                    res = parseInt(document.querySelector(\"#tab-label-reviews-title\")
                    .outerText.split(' ')[1]);
                    return res ? res : 0; }}
                )();
            """
        )
    except Exception:
        result = 0

    return result


async def shopping_get_rating_as_percentage(page: Page | PseudoPage) -> int:
    """Get the rating of the product on the shopping website as a percentage out of 100."""
    try:
        rating = await page.evaluate(
            """
                (() => {{
                    ratingPercentage = parseFloat(document.querySelector('.rating-result').title.replace('%', ''));
                    return ratingPercentage ? ratingPercentage : 0;
                }})();
            """
        )
    except Exception:
        rating = 0

    return rating


async def get_query_text(page: Page | PseudoPage, selector: str) -> str:
    """Get the text content of the element matching the given selector.

    Note that this function DOES NOT perform downcasing.
    """
    try:
        result = await page.evaluate(
            f"""
                (() => {{
                    try {{
                        return document.querySelector('{selector}').textContent;
                    }} catch (e) {{
                        return '';
                    }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


async def get_query_text_lowercase(page: Page | PseudoPage, selector: str) -> str:
    """Get the lowercase text content of the element matching the given selector."""
    text = await get_query_text(page, selector)
    return text.lower()


def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


async def reddit_get_post_comment_tree(page: Page | PseudoPage) -> Dict[str, Any]:
    try:
        comment_tree = await page.evaluate(
            f"""(function buildCommentTree(node, data_level) {{
    let tree = {{
        "username": node.querySelector(".fg-inherit").outerText,
        "net_score": parseInt(node.querySelector(".vote__net-score").outerText),
        "content": node.querySelector(".comment__content").outerText,
        "time": new Date(node.querySelector('.comment__main > header > h1 > span > time').dateTime),
        "children": []
    }};
    node.querySelectorAll(".comment").forEach((child) => {{
        if (parseInt(child.getAttribute('data-level')) === data_level+1) {{
            tree['children'].push(buildCommentTree(child, data_level+1));
        }}
    }})

    return tree;
}})(document.querySelector("#main"), 0)"""
        )
    except Exception:
        comment_tree = {}

    return comment_tree



async def reddit_get_latest_comment_obj_by_username(
    page: Page | PseudoPage, username: str
) -> Dict[str, Any]:
    try:
        comment_tree = await reddit_get_post_comment_tree(page)
        latest_time = datetime.min.replace(tzinfo=timezone.utc)
        comment = {}

        def dfs(node):
            nonlocal latest_time
            nonlocal comment
            if node["username"] == username:
                if node["time"] > latest_time:
                    comment = {
                        "username": node["username"],
                        "net_score": node["net_score"],
                        "content": node["content"],
                        "time": node["time"],
                    }
                    latest_time = node["time"]

            for child in node["children"]:
                dfs(child)

        dfs(comment_tree)

    except Exception as e:
        comment = {}
    return comment


async def reddit_get_latest_comment_content_by_username(
    page: Page | PseudoPage, username: str
) -> str:
    try:
        comment = await reddit_get_latest_comment_obj_by_username(page, username)
        content = comment["content"]

    except Exception:
        content = ""

    return content


async def reddit_get_parent_comment_obj_of_latest_comment_by_username(
    page: Page | PseudoPage, username: str
) -> Dict[str, Any]:
    try:
        comment_tree = await reddit_get_post_comment_tree(page)
        latest_time = datetime.min.replace(tzinfo=timezone.utc)
        comment = {}

        def dfs(node):
            nonlocal latest_time
            nonlocal comment
            for child in node["children"]:
                if child["username"] == username:
                    if child["time"] > latest_time:
                        comment = {
                            "username": node["username"],
                            "net_score": node["net_score"],
                            "content": node["content"],
                            "time": node["time"],
                        }
                        latest_time = child["time"]
                else:
                    dfs(child)

        dfs(comment_tree)

    except Exception:
        comment = {}
    return comment


async def reddit_get_parent_comment_username_of_latest_comment_by_username(
    page: Page | PseudoPage, username: str
) -> str:
    try:
        comment = await reddit_get_parent_comment_obj_of_latest_comment_by_username(
            page, username
        )
        username = comment["username"]

    except Exception:
        username = ""

    return username


async def gitlab_get_project_memeber_role(
    page: Page | PseudoPage, account_name: str
) -> str:
    # get the account index
    try:
        account_idx = await page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;  // Default value if not found

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )

        # get the role
        role: str = await page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        )
    except Exception:
        role = ""

    return role

@beartype
def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    # response = generate_from_openai_chat_completion(
    #     model="gpt-4-1106-preview",
    #     messages=messages,
    #     temperature=0,
    #     max_tokens=768,
    #     top_p=1.0,
    #     context_length=0,
    # ).lower()

    try:
        chat_completion = send_chat_completion_request_shangshu(messages=messages, temperature=0)
        if not chat_completion.choices or not chat_completion.choices[0].message.content:
            print("llm_fuzzy_match: Empty response from LLM")
            return 0.0
        response = chat_completion.choices[0].message.content.lower()
    except Exception as e:
        print(f"llm_fuzzy_match: Error calling LLM: {e}")
        return 0.0

    if "partially correct" in response or "incorrect" in response:
        return 0.0
    elif "correct" in response:
        return 1.0
    else:
        print(f"llm_fuzzy_match: Unexpected response format: {response}")
        return 0.0


def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    # response = generate_from_openai_chat_completion(
    #     model="gpt-4-1106-preview",
    #     messages=messages,
    #     temperature=0,
    #     max_tokens=768,
    #     top_p=1.0,
    #     context_length=0,
    # ).lower()

    try:
        chat_completion = send_chat_completion_request_shangshu(messages=messages, temperature=0)
        if not chat_completion.choices or not chat_completion.choices[0].message.content:
            print("llm_ua_match: Empty response from LLM")
            return 0.0
        response = chat_completion.choices[0].message.content.lower()
    except Exception as e:
        print(f"llm_ua_match: Error calling LLM: {e}")
        return 0.0

    if "different" in response:
        return 0.0
    elif "same" in response:
        return 1.0
    else:
        print(f"llm_ua_match: Unexpected response format: {response}")
        return 0.0


# 保留原有的常量定义
SCENE_NAME = "Qwen2_5_VL_72B_Instruct_vLLM_072_4L40S"
CHAIN_NAME = "v1"
MODEL_ENV = "prod"
API_VERSION = "v1"
OUT_SEQ_LENGTH = 1024
STREAM = False
REPETITION_PENALTY = 1.0
TEMPERATURE = 1.25
TOP_K = 50
TOP_P = 0.99
MAX_LENGTH = 4096
DO_SAMPLE = False

if not DO_SAMPLE:
    TEMPERATURE = 0.0
    TOP_K = -1
    TOP_P = 1.0

def get_streaming_response(response: requests.Response):
    for line in response.iter_lines(chunk_size=8192, decode_unicode=True):
        if line.startswith("data:"):
            json_data = line[len("data:"):].strip()
            if json_data == "[DONE]":
                return
            try:
                data = json.loads(json_data)
                content_chunk = data.get("data", None)
                if content_chunk:
                    yield content_chunk
            except json.JSONDecodeError:
                print("Error decoding JSON:", json_data)

def create_request_body(base64_image: str, prompt: str):
    data = {
        "api_version": API_VERSION,
        "out_seq_length": OUT_SEQ_LENGTH,
        "stream": STREAM,
        "prompts": [{
            "encoded_image": base64_image,
            "text": prompt,
            "repetition_penalty": REPETITION_PENALTY,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "max_length": MAX_LENGTH,
        }]
    }

    postman_data = {
        "sceneName": SCENE_NAME,
        "chainName": CHAIN_NAME,
        "modelEnv": MODEL_ENV,
        "isStream": STREAM,
        "itemId": "VL",
        "feature": data,
        "gwConfig": {"requestTimeout": 1800}
    }

    return json.dumps(postman_data)

def send_request(request_body, request_url, headers, max_retries=3, retry_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(request_url, headers=headers, data=request_body, timeout=120, stream=True)
            response.raise_for_status()  # 如果响应状态不是 200，将引发 HTTPError 异常

            if not STREAM:
                return response.json()['data']
            else:
                full_response = ""
                for content_chunk in get_streaming_response(response):
                    full_response += content_chunk
                return full_response

        except (RequestException, json.JSONDecodeError) as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed after {max_retries} attempts. Error: {str(e)}")
                return ""  # 重试失败后返回空字符串
            print(f"Request failed. Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2  # 指数退避策略

    # 这行代码理论上不会被执行到，因为在最后一次重试失败时会返回空字符串
    return ""

def qwen_vl_captioning_fn(images: List[Image.Image], prompts: List[str]) -> List[str]:
    request_url = os.environ.get("CODEBOT_ENDPOINT", "")
    headers = {
        "Content-Type": "application/json",
        'gpt_user': os.environ.get("CODEBOT_USER", ""),
        'gpt_token': os.environ.get("CODEBOT_TOKEN", ""),
    }

    results = []
    for image, prompt in zip(images, prompts):
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        request_body = create_request_body(img_str, prompt)
        result = send_request(request_body, request_url, headers)
        results.append(result)

    return results

# 使用示例
# images = [Image.open(requests.get(image_url, stream=True).raw) for image_url in image_urls]
# prompts = ["描述一下这张图片"] * len(images)
# results = captioning_fn(images, prompts)
# for result in results:
#     print(result)



if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你叫什么名字"},
    ]
    import time
    start_time = time.time()
    try:
        chat_completion = send_chat_completion_request_shangshu(messages=messages, temperature=0)
        qwen_str = chat_completion.choices[0].message.content
        print("qwen_str: ", qwen_str)
    except Exception as e:
        print(f"Error: {e}")
    print("end time: ", time.time() - start_time)


