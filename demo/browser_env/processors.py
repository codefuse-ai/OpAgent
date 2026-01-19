import json
import asyncio
import logging
import pkgutil
import re
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any, Optional, TypedDict, Union, Dict
from urllib.parse import urljoin, urlparse
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import playwright
import requests
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import CDPSession, Page, ViewportSize
from playwright.sync_api import Error as PlaywrightError
from playwright.async_api import BrowserContext as ABrowserContext
from playwright.async_api import Locator as ALocator
from playwright.async_api import Page as APage
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
import uuid
import base64
from .constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    IGNORED_ACTREE_PROPERTIES,
    INJECTED_ATTR_NAME,
    UTTERANCE_MAX_LENGTH,
    BID_ATTR,
    DATA_REGEXP,
    IN_VIEWPORT_RATIO_THRESHOLD,
)

from .utils import (
    AccessibilityTree,
    AccessibilityTreeNode,
    BrowserConfig,
    BrowserInfo,
    DOMNode,
    DOMTree,
    Observation,
    png_bytes_to_numpy,
)
from datetime import datetime
import os
from .utils import with_timeout_legacy

# --- Logger Setup ---
# 配置logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
# 为了能看到所有级别的日志，您可能需要在您的主程序中配置logging
# 例如: logging.basicConfig(level=logging.INFO)

VLM_EXP_DEBUG = os.environ.get('VLM_EXP_DEBUG', '0')
VLM_EXP_DEBUG = str(VLM_EXP_DEBUG)
VLM_EXP_FONT_PATH = os.environ.get('VLM_EXP_FONT_PATH', "./media/SourceCodePro-SemiBold.ttf")
EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', "")

logger.info(f"VLM_EXP_FONT_PATH set to: {VLM_EXP_FONT_PATH}")


def remove_unicode(input_string):
    # Define a regex pattern to match Unicode characters
    unicode_pattern = re.compile(r"[^\x00-\x7F]+")

    # Use the pattern to replace Unicode characters with an empty string
    cleaned_string = unicode_pattern.sub("", input_string)

    return cleaned_string


class ObservationProcessor:
    def process(self, page: Page) -> Observation:
        raise NotImplementedError


class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]


def create_empty_metadata() -> ObservationMetadata:
    return {
        "obs_nodes_info": {},
    }


def extract_data_items_from_aria(string: str) -> tuple[list[str], str]:
    """
    Utility function to extract temporary data stored in the "aria-roledescription" attribute of a node
    """

    match = DATA_REGEXP.fullmatch(string)
    if not match:
        return [], string

    groups = match.groups()
    data_items = groups[:-1]
    original_aria = groups[-1]
    return data_items, original_aria


class TextObervationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ):
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size
        self.observation_tag = "text"
        self.meta_data = (
            create_empty_metadata()
        )  # use the store meta data of this observation type

        if self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            self.captioning_fn = captioning_fn
            # Cache captions.
            self.url2caption = {}

    def fetch_browser_info(
        self,
        page: Page,
    ) -> BrowserInfo:
        # extract domtree
        client = page.context.new_cdp_session(page)
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )
        client.detach()

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    @staticmethod
    def get_bounding_client_rect(
        client: CDPSession, backend_node_id: str
    ) -> dict[str, Any]:
        try:
            remote_object = client.send(
                "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
            )
            remote_object_id = remote_object["object"]["objectId"]
            response = client.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": remote_object_id,
                    "functionDeclaration": """
                        function() {
                            if (this.nodeType == 3) {
                                var range = document.createRange();
                                range.selectNode(this);
                                var rect = range.getBoundingClientRect().toJSON();
                                range.detach();
                                return rect;
                            } else {
                                return this.getBoundingClientRect().toJSON();
                            }
                        }
                    """,
                    "returnByValue": True,
                },
            )
            return response
        except Exception as e:
            return {"result": {"subtype": "error"}}

    @staticmethod
    def get_element_in_viewport_ratio(
        elem_left_bound: float,
        elem_top_bound: float,
        width: float,
        height: float,
        config: BrowserConfig,
    ) -> float:
        elem_right_bound = elem_left_bound + width
        elem_lower_bound = elem_top_bound + height

        win_left_bound = 0
        win_right_bound = config["win_width"]
        win_top_bound = 0
        win_lower_bound = config["win_height"]

        # Compute the overlap in x and y axes
        overlap_width = max(
            0,
            min(elem_right_bound, win_right_bound)
            - max(elem_left_bound, win_left_bound),
        )
        overlap_height = max(
            0,
            min(elem_lower_bound, win_lower_bound) - max(elem_top_bound, win_top_bound),
        )

        # Compute the overlap area
        ratio = overlap_width * overlap_height / width * height
        return ratio

    def fetch_page_html(
        self,
        info: BrowserInfo,
        page: Page,
        current_viewport_only: bool,
    ) -> DOMTree:
        # adopted from [natbot](https://github.com/nat/natbot)
        tree = info["DOMTree"]
        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]

        # make a dom tree that is easier to navigate
        dom_tree: DOMTree = []
        graph = defaultdict(list)
        client = page.context.new_cdp_session(page)
        for node_idx in range(len(nodes["nodeName"])):
            cur_node: DOMNode = {
                "nodeId": "",
                "nodeType": "",
                "nodeName": "",
                "nodeValue": "",
                "attributes": "",
                "backendNodeId": "",
                "parentId": "",
                "childIds": [],
                "cursor": 0,
                "union_bound": None,
            }

            node_type_idx = nodes["nodeType"][node_idx]
            node_type = "generic"
            if node_type_idx >= 0 and node_type_idx < len(strings):
                node_type = strings[node_type_idx]

            node_name = strings[nodes["nodeName"][node_idx]]

            node_value_idx = nodes["nodeValue"][node_idx]
            node_value = ""
            if node_value_idx >= 0 and node_value_idx < len(strings):
                node_value = " ".join(strings[node_value_idx].split())

            node_attributes = [strings[i] for i in nodes["attributes"][node_idx]]
            node_attributes_str = ""
            for i in range(0, len(node_attributes), 2):
                a = node_attributes[i]
                b = node_attributes[i + 1]
                b = " ".join(b.split())
                node_attributes_str += f'{a}="{b}" '
            node_attributes_str = node_attributes_str.strip()

            cur_node["nodeId"] = str(node_idx)
            cur_node["nodeType"] = node_type
            cur_node["nodeName"] = node_name
            cur_node["nodeValue"] = node_value
            cur_node["attributes"] = node_attributes_str
            cur_node["backendNodeId"] = str(nodes["backendNodeId"][node_idx])
            cur_node["parentId"] = str(nodes["parentIndex"][node_idx])

            if cur_node["parentId"] != "-1":
                graph[cur_node["parentId"]].append(str(cur_node["nodeId"]))

            # get the bound
            if cur_node["parentId"] == "-1":
                cur_node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(
                    client, cur_node["backendNodeId"]
                )
                if response.get("result", {}).get("subtype", "") == "error":
                    cur_node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    cur_node["union_bound"] = [x, y, width, height]

            dom_tree.append(cur_node)

        client.detach()
        # add parent children index to the node
        for parent_id, child_ids in graph.items():
            dom_tree[int(parent_id)]["childIds"] = child_ids

        # remove the nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node: DOMNode) -> None:
                # update the node information in the accessibility tree
                node_id = node["nodeId"]
                parent_id = node["parentId"]
                child_ids = node["childIds"]

                # update the children of the parent node
                assert dom_tree[int(parent_id)]["parentId"] != "[REMOVED]"
                # remove the nodeid from parent
                index = dom_tree[int(parent_id)]["childIds"].index(node_id)
                dom_tree[int(parent_id)]["childIds"].pop(index)

                # Insert children_nodeids in the same location
                for child_id in child_ids:
                    dom_tree[int(parent_id)]["childIds"].insert(index, child_id)
                    index += 1

                # update children node's parent
                for child_id in child_ids:
                    dom_tree[int(child_id)]["parentId"] = parent_id
                # mark as removed
                dom_tree[int(node_id)]["parentId"] = "[REMOVED]"

            config = info["config"]
            for cursor, node in enumerate(dom_tree):
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0.0 or height == 0.0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            dom_tree = [
                node for node in dom_tree if node.get("parentId", "-1") != "[REMOVED]"
            ]

        return dom_tree

    @staticmethod
    def parse_html(dom_tree: DOMTree) -> tuple[str, dict[str, Any]]:
        """Parse the html tree into a string text"""

        obs_nodes_info = {}
        nodeid_to_cursor = {node["nodeId"]: idx for idx, node in enumerate(dom_tree)}

        def dfs(node_cursor: int, depth: int) -> str:
            tree_str = ""
            node = dom_tree[node_cursor]
            indent = "\t" * depth
            valid_node = True
            try:
                node_str = f"[{node_cursor}] <{node['nodeName']}"
                if node["attributes"]:
                    node_str += f" {node['attributes']}"
                node_str += f"> {node['nodeValue']}"
                valid_node = bool(node["attributes"] or node["nodeValue"])

                if valid_node:
                    obs_nodes_info[str(node_cursor)] = {
                        "backend_id": node["backendNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }
                    tree_str += f"{indent}{node_str}\n"

            except Exception as e:
                valid_node = False

            for child_ids in node["childIds"]:
                child_cursor = nodeid_to_cursor[child_ids]
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(child_cursor, child_depth)
                tree_str += child_str

            return tree_str

        html = dfs(0, 0)
        return html, obs_nodes_info

    def fetch_page_accessibility_tree(
        self,
        page: Page,
        info: BrowserInfo,
        current_viewport_only: bool,
    ) -> AccessibilityTree:
        client = page.context.new_cdp_session(page)
        accessibility_tree: AccessibilityTree = client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]

        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        nodeid_to_cursor = {}
        for cursor, node in enumerate(accessibility_tree):
            nodeid_to_cursor[node["nodeId"]] = cursor
            # usually because the node is not visible etc
            if "backendDOMNodeId" not in node:
                node["union_bound"] = None
                continue
            backend_node_id = str(node["backendDOMNodeId"])
            if node["role"]["value"] == "RootWebArea":
                # always inside the viewport
                node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(
                    client,
                    backend_node_id
                )
                if response.get("result", {}).get("subtype", "") == "error":
                    node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    node["union_bound"] = [x, y, width, height]

        client.detach()
        # filter nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
                # update the node information in the accessibility tree
                nodeid = node["nodeId"]
                node_cursor = nodeid_to_cursor[nodeid]
                parent_nodeid = node["parentId"]
                children_nodeids = node["childIds"]
                parent_cursor = nodeid_to_cursor[parent_nodeid]
                # update the children of the parent node
                assert (
                    accessibility_tree[parent_cursor].get("parentId", "Root")
                    is not None
                )
                # remove the nodeid from parent's childIds
                index = accessibility_tree[parent_cursor]["childIds"].index(nodeid)
                accessibility_tree[parent_cursor]["childIds"].pop(index)
                # Insert children_nodeids in the same location
                for child_nodeid in children_nodeids:
                    accessibility_tree[parent_cursor]["childIds"].insert(
                        index, child_nodeid
                    )
                    index += 1
                # update children node's parent
                for child_nodeid in children_nodeids:
                    child_cursor = nodeid_to_cursor[child_nodeid]
                    accessibility_tree[child_cursor]["parentId"] = parent_nodeid
                # mark as removed
                accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

            config = info["config"]
            for node in accessibility_tree:
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0 or height == 0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            accessibility_tree = [
                node
                for node in accessibility_tree
                if node.get("parentId", "Root") != "[REMOVED]"
            ]

        return accessibility_tree

    @staticmethod
    def parse_accessibility_tree(
        accessibility_tree: AccessibilityTree,
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }

            except Exception as e:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree_str, obs_nodes_info

    @staticmethod
    def clean_accesibility_tree(tree_str: str) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            # remove statictext if the content already appears in the previous line
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                pattern = r"\[\d+\] StaticText (.+)"

                match = re.search(pattern, line, re.DOTALL)
                if match:
                    static_text = match.group(1)[1:-1]  # remove the quotes
                    if static_text and all(
                        static_text not in prev_line for prev_line in prev_lines
                    ):
                        clean_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines)

    def fetch_image_related(self, page: Page, browser_info: BrowserInfo) -> str:
        # Check if the current page is an image url
        if page.url.endswith((".jpg", ".jpeg", ".png")):
            logger.info("Current page is an image URL.")
            # Load image from current url and run captioning on it.
            if page.url not in self.url2caption and self.captioning_fn is not None:
                try:
                    image = Image.open(requests.get(page.url, stream=True).raw)
                    caption = self.captioning_fn([image])[0].strip()
                    self.url2caption[page.url] = remove_unicode(caption)
                except Exception as e:
                    logger.warning("Failed to caption image from URL %s: %s", page.url, e)
            content = self.url2caption.get(page.url, "Image")

        else:
            if self.captioning_fn is not None:
                images = page.query_selector_all("img")
                image_urls = []
                for image in images:
                    try:
                        image_url = image.get_attribute("src")
                        if not image_url.startswith(("http://", "https://", "www.")):
                            image_url = urljoin(page.url, image_url)
                        if image_url not in self.url2caption:
                            image_urls.append(image_url)
                    except Exception as e:
                        logger.warning("Failed to get 'src' attribute from an image element: %s", e)

                # Run image captioning on image_url pixels. This is for models which use captioning as a baseline.
                if len(image_urls) > 0:
                    image_pixels = []
                    valid_urls = []
                    for url in image_urls:
                        if "data:image/svg" in url:
                            continue
                        else:
                            try:
                                image = Image.open(requests.get(url, stream=True).raw)
                                image_pixels.append(image)
                                valid_urls.append(url)
                            except Exception as e:
                                logger.warning("Failed to open image from URL %s: %s", url, e)

                    # Caption images.
                    if image_pixels:
                        # Run in batches of 4.
                        bs = 4
                        captions = []
                        for i in range(0, len(image_pixels), bs):
                            try:
                                captions.extend(
                                    self.captioning_fn(image_pixels[i : i + bs])
                                )
                            except Exception as e:
                                logger.warning("Captioning function failed for a batch of images: %s", e)
                                captions.extend([""] * len(image_pixels[i : i + bs]))
                        assert len(valid_urls) == len(
                            captions
                        ), f"len(images)={len(valid_urls)}, len(captions)={len(captions)}"
                        for image_url, caption in zip(valid_urls, captions):
                            self.url2caption[image_url] = remove_unicode(
                                caption.strip()
                            )

                image_idx = 0
                for image in images:
                    try:
                        original_alt = image.get_attribute("alt") or ""
                        image_url = image.get_attribute("src")
                        if not image_url.startswith(("http://", "https://", "www.")):
                            image_url = urljoin(page.url, image_url)

                        updated_alt = original_alt

                        if image_url in self.url2caption:
                            if self.url2caption[image_url] not in updated_alt:
                                updated_alt = f"{updated_alt}, description: {self.url2caption[image_url]}"
                        elif "data:image/svg" not in image_url:
                            logger.warning("Image URL %s not found in caption cache.", image_url)

                        if "url:" not in updated_alt:
                            updated_alt = f"{updated_alt}, url: {image_url}"

                        safe_updated_alt = json.dumps(updated_alt)
                        image.evaluate(f"node => node.alt = {safe_updated_alt}")
                    except Exception as e:
                        logger.warning("Failed to update alt text for an image element: %s", e)

            if self.observation_type == "accessibility_tree_with_captioner":
                frame_ax_trees = self.fetch_page_accessibility_tree(
                    page,
                    browser_info,
                    current_viewport_only=self.current_viewport_only
                )
                content, obs_nodes_info = self.parse_accessibility_tree(frame_ax_trees)
                content = self.clean_accesibility_tree(content)
                self.obs_nodes_info = obs_nodes_info
                self.meta_data["obs_nodes_info"] = obs_nodes_info
            else:
                content = ""  # Not used for SoM

        return content

    def process(self, page: Page) -> str:
        # get the tab info
        open_tabs = page.context.pages
        try:
            tab_titles = [tab.title() for tab in open_tabs]
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[idx] = f"Tab {idx} (current): {open_tabs[idx].title()}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {open_tabs[idx].title()}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join([f"Tab {idx}" for idx in range(len(open_tabs))])

        try:
            browser_info = self.fetch_browser_info(page)
        except Exception:
            page.wait_for_load_state("load", timeout=10 * 60 * 1000)
            browser_info = self.fetch_browser_info(page)

        if self.observation_type == "html":
            dom_tree = self.fetch_page_html(
                browser_info,
                page,
                self.current_viewport_only,
            )
            content, obs_nodes_info = self.parse_html(dom_tree)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info

        elif self.observation_type == "accessibility_tree":
            accessibility_tree = self.fetch_page_accessibility_tree(
                page,
                browser_info,
                self.current_viewport_only,
            )
            content, obs_nodes_info = self.parse_accessibility_tree(accessibility_tree)
            content = self.clean_accesibility_tree(content)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info

        elif self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            content = self.fetch_image_related(
                page,
                browser_info,
            )

        elif self.observation_type == "":
            content = ""

        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        self.browser_config = browser_info["config"]
        content = f"{tab_title_str}\n\n{content}"

        return content

    def get_element_center(self, element_id: str) -> tuple[float, float]:
        node_info = self.obs_nodes_info[element_id]
        node_bound = node_info["union_bound"]
        x, y, width, height = node_bound
        center_x = x + width / 2
        center_y = y + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )


class ImageObservationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        viewport_size: Optional[ViewportSize] = None,
    ):
        self.observation_type = observation_type
        self.observation_tag = "image"
        self.viewport_size = viewport_size
        self.meta_data = create_empty_metadata()

    def get_page_bboxes(self, page: Page) -> list[list[float]]:
        """JavaScript code to return bounding boxes and other metadata from HTML elements."""
        js_script = """
        (() => {
            const interactableSelectors = [
                'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
                '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
                '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]',
                '.btn', 'a[href="/notifications"]', 'a[href="/submit"]', '.fa.fa-star.is-rating-item', 'input[type="checkbox"]'

            ];

            const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
            const modifiedTextSelectors = textSelectors.map(selector =>
                `:not(${interactableSelectors.join(', ')}):not(style) > ${selector}`
            );

            const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
            const elements = document.querySelectorAll(combinedSelectors.join(', '));

            const pixelRatio = window.devicePixelRatio;
            let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable\\n";
            let counter = 1;

            elements.forEach(element => {
                const rect = element.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                let altText = element.getAttribute('alt') || '';
                altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
                const classList = element.className || '';
                const id = element.id || '';
                let textContent = element.textContent || '';
                textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent

                // Determine if the element is interactable
                const isInteractable = interactableSelectors.some(selector => element.matches(selector));

                const dataString = [
                    counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
                    (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                    (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                    altText, classList, id, textContent, isInteractable
                ].map(value => `"${value}"`).join(",");

                csvContent += dataString + "\\n";
                counter++;
            });

            return csvContent;
        })();
        """
        # Save the bbox as a CSV
        csv_content = page.evaluate(js_script)
        return csv_content

    def draw_bounding_boxes(
        self,
        data_string,
        screenshot_img,
        viewport_size=None,
        add_ids=True,
        bbox_color=None,
        min_width=8,
        min_height=8,
        bbox_padding=0,
        bbox_border=2,
        plot_ids=None,
    ):
        """
        min_width and min_height: Minimum dimensions of the bounding box to be plotted.
        """
        # 1. 使用原生json库，更快、更轻
        try:
            elements = json.loads(data_string)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON data.")
            return screenshot_img, {}, "" # 返回空结果

        img = screenshot_img.copy()
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(VLM_EXP_FONT_PATH, 16)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        id2center = {}
        text_content_elements = []
        
        # 注意：这里我们假设JS已经过滤了视口外的元素
        # 并且已经处理好了坐标，这里不再需要减去 b_x, b_y
        # 如果JS返回的是文档坐标，而截图是视口截图，则仍需转换

        for element in elements:
            unique_id = str(element['id'])
            
            if element['interactable']:
                left, top, right, bottom = element['bbox_pixels']
                width = right - left
                height = bottom - top

                if width < min_width or height < min_height:
                    continue

                # 存储中心点信息
                id2center[unique_id] = (
                    (left + right) / 2,
                    (bottom + top) / 2,
                    width,
                    height,
                    True
                )

                # 绘制方框和ID
                color = bbox_color or color_cycle[int(unique_id) % len(color_cycle)]
                draw.rectangle([left, top, right, bottom], outline=color, width=bbox_border)
                
                # (这里的智能标签放置逻辑可以保留，因为它很优秀)
                # 为了简化示例，这里只做一个简单的绘制
                text_pos = (left, top - 18) # 简单放在左上角外部
                draw.rectangle([text_pos[0], text_pos[1], text_pos[0] + 20, text_pos[1] + 18], fill=color)
                draw.text(text_pos, unique_id, font=font, fill="white")

                # 添加到文本描述
                content = element['name']
                text_content_elements.append(f"[{unique_id}] [{element['tag']}] [{content}]")

            else: # 如果是静态文本
                content = element['name']
                # 可以选择性地将非交互文本也加入描述，如果需要的话
                # text_content_elements.append(f"[StaticText] [{content}]")

        content_str = "\n".join(text_content_elements)
        logger.debug(f"draw_bounding_boxes_optimized finished")
        
        return img, id2center, content_str

    def rectangles_overlap(self, rect1, rect2, padding):
        """
        Check if two rectangles overlap.
        Each rectangle is represented as a list [x1, y1, x2, y2].
        """
        return not (
            rect1[2] < rect2[0] + padding
            or rect1[0] > rect2[2] - padding
            or rect1[1] > rect2[3] - padding
            or rect1[3] < rect2[1] + padding
        )

    def process(self, page: Page) -> npt.NDArray[np.uint8]:
        try:
            browser_info = self.fetch_browser_info(page)
        except Exception:
            page.wait_for_load_state("load", timeout=10 * 60 * 1000)
            browser_info = self.fetch_browser_info(page)

        self.browser_config = browser_info["config"]

        if self.observation_type == "image_som":
            # Produce the SoM image, with bounding boxes
            try:
                screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page)
                screenshot_img = Image.open(BytesIO(screenshot_bytes))
                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                screenshot_som = np.array(bbox_img)
                return screenshot_som, content_str
            except:
                page.wait_for_event("load")
                screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page)
                screenshot_img = Image.open(BytesIO(screenshot_bytes))
                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                screenshot_som = np.array(bbox_img)
                return screenshot_som, content_str
        else:
            try:
                screenshot = png_bytes_to_numpy(page.screenshot())
            except:
                page.wait_for_event("load")
                screenshot = png_bytes_to_numpy(page.screenshot())
            return screenshot, ""

    # ### 改动：fetch_browser_info 引入了更健壮的错误处理和超时 ###
    def fetch_browser_info(self, page: Page) -> Dict[str, Any]: # 使用 Dict[str, Any] 替代 BrowserInfo 以便演示
        """
        获取浏览器信息（DOM、视口等）。通过重试机制加固，以处理页面导航的竞争条件。
        这是同步版本。
        """
        # --- 默认的错误返回值 ---
        default_error_return = {
            "DOMTree": None,
            "config": {
                "win_upper_bound": 0, "win_left_bound": 0,
                "win_width": self.viewport_size.get("width", 1920),
                "win_height": self.viewport_size.get("height", 1080),
                "device_pixel_ratio": 1.0,
                "error": "An error occurred"
            }
        }

        # --- 重试配置 ---
        max_retries = 3
        retry_delay_seconds = 0.5  # 每次重试前等待500毫秒

        for attempt in range(max_retries):
            client: CDPSession | None = None
            try:
                # 如果页面已经关闭，尝试获取信息没有意义
                if page.is_closed():
                    logger.warning("试图从一个已关闭的页面获取信息。")
                    return default_error_return

                client = page.context.new_cdp_session(page)

                # 获取 DOM 快照
                tree = client.send(
                    "DOMSnapshot.captureSnapshot",
                    {"computedStyles": [], "includeDOMRects": True, "includePaintOrder": True}
                )

                # 通过 JavaScript evaluation 获取页面指标
                win_upper_bound = page.evaluate("window.pageYOffset")
                win_left_bound = page.evaluate("window.pageXOffset")
                win_width = page.evaluate("window.screen.width")
                win_height = page.evaluate("window.screen.height")
                device_pixel_ratio = page.evaluate("window.devicePixelRatio")

                # --- 成功情况 ---
                # 如果所有操作都成功，分离客户端并处理数据
                client.detach()
                client = None  # 标记为已分离

                # 数据处理部分 (与您原始逻辑相同，但增加了健壮性检查)
                try:
                    bounds = tree["documents"][0]["layout"]["bounds"]
                    if bounds:
                        # 确保 viewport_size["width"] > 0
                        viewport_width = self.viewport_size.get("width", 0)
                        if viewport_width > 0:
                            scale = bounds[0][2] / viewport_width
                            if scale > 0:
                                bounds = [[x / scale for x in bound] for bound in bounds]
                        tree["documents"][0]["layout"]["bounds"] = bounds
                except (IndexError, KeyError, ZeroDivisionError) as proc_err:
                    logger.warning(f"处理DOM快照的bounds时出错: {proc_err}")
                    # 即使处理失败，我们仍然可以返回获取到的其他信息
                    pass

                config = {
                    "win_upper_bound": win_upper_bound,
                    "win_left_bound": win_left_bound,
                    "win_width": win_width,
                    "win_height": win_height,
                    "device_pixel_ratio": device_pixel_ratio,
                }

                # 返回成功的结果并退出函数
                return {"DOMTree": tree, "config": config}

            except PlaywrightError as e:
                # --- 循环内的错误处理 ---
                error_message = str(e)
                # 检查是否是我们期望的导航错误
                if "Execution context was destroyed" in error_message or \
                   "Target closed" in error_message:
                    # 这就是关键：我们捕获到了导航竞争条件。
                    logger.warning(
                        f"尝试 {attempt + 1}/{max_retries}: "
                        f"在获取信息时检测到页面跳转。将在 {retry_delay_seconds} 秒后重试..."
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay_seconds)
                        # 循环将继续下一次尝试
                        continue
                    else:
                        # 这是最后一次尝试
                        logger.error("已达到最大重试次数。由于反复的页面跳转，获取浏览器信息失败。")
                        default_error_return["config"]["error"] = "已达到最大重试次数。执行上下文被反复销毁。"
                        break  # 退出循环，以返回默认错误
                else:
                    # 这是其他未预料到的 Playwright 错误（例如，超时）。
                    # 我们不应该为这些错误重试。立即记录日志并失败。
                    logger.error(f"在 fetch_browser_info 中发生未预料的 Playwright 错误: {e}", exc_info=True)
                    default_error_return["config"]["error"] = f"未预料的 Playwright 错误: {error_message}"
                    break  # 退出循环，以返回默认错误
            except Exception as e:
                # 捕获其他非Playwright的通用异常
                logger.error(f"在 fetch_browser_info 中发生未知错误: {e}", exc_info=True)
                default_error_return["config"]["error"] = f"未知错误: {str(e)}"
                break
            finally:
                # 确保 CDP 客户端总是被分离，即使发生异常
                if client:
                    try:
                        client.detach()
                    except PlaywrightError as detach_error:
                        # 如果页面已关闭，这里可能会出错，可以安全地忽略
                        logger.warning(f"在清理CDP客户端时忽略错误: {detach_error}")

        # 如果循环结束都没有成功返回，则返回默认的错误信息
        return default_error_return

    def get_element_center(self, element_id: str) -> tuple[float, float]:
        if not self.observation_type == "image_som":
            raise ValueError(
                "get_element_center() is only supported for 'image_som' observation type."
            )

        browser_config = self.browser_config
        center_x, center_y, width, height = self.som_id_info[element_id]
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )


class ObservationHandler:
    """Main entry point to access all observation processor"""

    def __init__(
        self,
        main_observation_type: str,
        text_observation_type: str,
        image_observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,

    ) -> None:
        self.main_observation_type = main_observation_type
        self.text_processor = TextObervationProcessor(
            text_observation_type,
            current_viewport_only,
            viewport_size,
            captioning_fn,
        )
        self.image_processor = ImageObservationProcessor(
            image_observation_type, viewport_size
        )
        self.viewport_size = viewport_size

    def get_observation_space(self) -> spaces.Dict:
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )

        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )

        return spaces.Dict({"text": text_space, "image": image_space})

    def get_observation(self, page: Page) -> dict[str, Observation]:
        text_obs = self.text_processor.process(page)
        image_obs, content_str = self.image_processor.process(page)
        if content_str != "":
            text_obs = content_str
        return {"text": text_obs, "image": image_obs}

    def get_observation_metadata(self) -> dict[str, ObservationMetadata]:
        return {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }

    @property
    def action_processor(self) -> ObservationProcessor:
        """Return the main processor that is associated with the action space"""
        if self.main_observation_type == "text":
            return self.text_processor
        elif self.main_observation_type == "image":
            return self.image_processor
        else:
            raise ValueError("Invalid main observation type")


class AsyncImageObservationProcessor(ImageObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        viewport_size: ViewportSize,
        context_id: str = "",
    ):
        self.observation_type = observation_type
        self.observation_tag = "image"
        self.viewport_size = viewport_size
        self.meta_data = self.create_empty_metadata()
        self.som_id_info = {}
        self.save_img_file = ""
        self.context_id = context_id

    def get_observation_space(self):
        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )
        return image_space

    def get_observation_metadata(self):
        return self.meta_data

    async def async_get_observation(self, apage: APage) -> tuple[npt.NDArray[np.uint8], str]:
        image_obs, content_str = await self.async_process(apage)
        logger.debug(f"context_id  {self.context_id} url {apage.url} out async_get_observation")
        return image_obs, content_str

    def create_empty_metadata(self) -> ObservationMetadata:
        return {
            "obs_nodes_info": {}
        }

    @with_timeout_legacy(2 * 60)
    async def async_get_page_bboxes(self, apage: APage) -> str:
        js_script = """
        (() => {
            // 1. 更简单、更广泛的选择器
            const selectors = [
                'a', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
                '[role]', '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]',
                'p', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'
            ];
            const elements = document.querySelectorAll(selectors.join(', '));
            const pixelRatio = window.devicePixelRatio;
            const results = [];
            let counter = 0;

            // 辅助函数：检查元素是否在视口内
            function isInViewport(rect) {
                return (
                    rect.top >= 0 &&
                    rect.left >= 0 &&
                    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
                );
            }

            // 辅助函数：获取元素的"name"
            function getElementName(element) {
                // 优先使用aria-label，这是最干净的语义标签
                let name = element.getAttribute('aria-label') || element.getAttribute('alt') || element.innerText || element.value || '';
                // 清理并截断文本，防止过长
                return name.replace(/\\s+/g, ' ').trim().substring(0, 100);
            }

            elements.forEach(element => {
                const rect = element.getBoundingClientRect();

                // 2. 关键优化：过滤不可见和视口外的元素
                if (rect.width < 1 || rect.height < 1 || !isInViewport(rect)) {
                    return;
                }

                // 3. 关键优化：遮挡检查
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                const elementAtCenter = document.elementFromPoint(centerX, centerY);
                if (elementAtCenter && !element.contains(elementAtCenter)) {
                    return; // 元素被其他东西遮挡
                }

                // 4. 在循环中判断是否可交互，更灵活
                const isInteractable = window.getComputedStyle(element).cursor === 'pointer' ||
                                    element.hasAttribute('onclick') ||
                                    ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName) ||
                                    (element.hasAttribute('role') && ['button', 'link', 'checkbox', 'menuitem', 'tab'].includes(element.getAttribute('role')));


                results.push({
                    id: counter++,
                    tag: element.tagName,
                    name: getElementName(element),
                    bbox_pixels: [
                        rect.left * pixelRatio,
                        rect.top * pixelRatio,
                        rect.right * pixelRatio,
                        rect.bottom * pixelRatio
                    ],
                    interactable: isInteractable
                });
            });

            // 5. 关键优化：返回JSON字符串
            return JSON.stringify(results);
        })();
        """
        return await apage.evaluate(js_script)


    @with_timeout_legacy(2 * 60)
    async def async_fetch_browser_info(self, apage: APage) -> Dict[str, Any]:
        logger.debug(f"in async_fetch_browser_info")
        # 定义一个合理的超时（秒）
        # 注意：10 * 60.0 是10分钟，对于一个单一操作来说太长了。
        # 这里我仍然强烈建议使用一个更短的时间，比如 15.0 (15秒)
        # 如果你确定需要长超时，可以保留，但请意识到它可能带来的性能影响。
        CDP_TIMEOUT_SECONDS = 10 * 60.0

        # 准备一个默认的、表示失败的返回值结构
        default_error_return = {
            "DOMTree": None,
            "config": {
                "win_upper_bound": 0,
                "win_left_bound": 0,
                "win_width": self.viewport_size.get("width", 1920),
                "win_height": self.viewport_size.get("height", 1080),
                "device_pixel_ratio": 1,
                "error": "An error occurred"
            }
        }

        # --- 重试配置 ---
        max_retries = 3      # 最多重试3次
        retry_delay_seconds = 0.5  # 每次重试前等待500毫秒

        for attempt in range(max_retries):
            client: CDPSession | None = None
            try:
                # 如果页面已经关闭，尝试获取信息没有意义
                if apage.is_closed():
                    logger.warning("试图从一个已关闭的页面获取信息。")
                    return default_error_return

                context = apage.context
                client = await context.new_cdp_session(apage)
                logger.debug(f"await client")
                # 获取 DOM 快照
                tree = await client.send(
                    "DOMSnapshot.captureSnapshot",
                    {
                        "computedStyles": [],
                        "includeDOMRects": True,
                        "includePaintOrder": True,
                    },
                )

                # 通过 JavaScript evaluation 获取页面指标
                win_upper_bound = await apage.evaluate("window.pageYOffset")
                win_left_bound = await apage.evaluate("window.pageXOffset")
                win_width = await apage.evaluate("window.screen.width")
                win_height = await apage.evaluate("window.screen.height")
                device_pixel_ratio = await apage.evaluate("window.devicePixelRatio")

                # --- 成功情况 ---
                # 如果所有操作都成功，分离客户端并处理数据
                await client.detach()
                client = None # 标记为已分离

                # 处理 DOM 树的 bounds (与您原始逻辑相同)
                bounds = tree["documents"][0]["layout"]["bounds"]
                scale = bounds[0][2] / self.viewport_size["width"] if self.viewport_size else 1
                if scale > 0:
                    bounds = [[x / scale for x in bound] for bound in bounds]
                tree["documents"][0]["layout"]["bounds"] = bounds

                config = {
                    "win_upper_bound": win_upper_bound,
                    "win_left_bound": win_left_bound,
                    "win_width": win_width,
                    "win_height": win_height,
                    "device_pixel_ratio": device_pixel_ratio,
                }
                logger.debug(f"async_fetch_browser_info finished")
                # 返回成功的结果并退出函数
                return {"DOMTree": tree, "config": config}

            except Exception as e:
                # --- 循环内的错误处理 ---
                error_message = str(e)
                # 检查是否是我们期望的导航错误
                if "Execution context was destroyed" in error_message or \
                   "Target closed" in error_message:
                    # 这就是关键：我们捕获到了导航竞争条件。
                    logger.warning(
                        f"尝试 {attempt + 1}/{max_retries}: "
                        f"在获取信息时检测到页面跳转。将在 {retry_delay_seconds} 秒后重试..."
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay_seconds)
                        # 循环将继续下一次尝试
                        continue
                    else:
                        # 这是最后一次尝试
                        logger.error("已达到最大重试次数。由于反复的页面跳转，获取浏览器信息失败。")
                        default_error_return["config"]["error"] = "已达到最大重试次数。执行上下文被反复销毁。"
                        break # 退出循环，以返回默认错误
                else:
                    # 这是其他未预料到的错误（例如，超时、协议错误）。
                    # 我们不应该为这些错误重试。立即记录日志并失败。
                    logger.error(f"在 async_fetch_browser_info 中发生意外错误: {e}", exc_info=True)
                    default_error_return["config"]["error"] = f"意外错误: {error_message}"
                    break # 退出循环，以返回默认错误
            finally:
                # 确保 CDP 客户端总是被分离，即使发生异常
                if client:
                    try:
                        await client.detach()
                    except Exception as detach_error:
                        # 如果页面已关闭，这里可能会出错，可以安全地忽略
                        logger.warning(f"在清理CDP客户端时忽略错误: {detach_error}")

        # 如果循环结束都没有成功返回，则返回默认的错误信息
        return default_error_return

    def save_image(self, bbox_img, save_name):
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        timestamp_str = str(uuid.uuid4())
        # TODO: 配置您的临时文件路径
        tmp_dir = os.getenv("TMP_DIR", "./tmp")
        self.save_img_file = f"{tmp_dir}/{save_name}_{timestamp_str}.png"
        logger.info(f"Saving bounding box image to: {self.save_img_file}")
        bbox_img.save(self.save_img_file)
        return

    async def async_process(self, apage: APage) -> tuple[np.ndarray, str]:
        """
        健壮的观察数据获取函数，包含了对 async_fetch_browser_info 的安全重试逻辑。
        """
        logger.debug(f"in async_process url {apage.url}")
        DEFAULT_TIMEOUT = 2 * 60 * 1000  # 30秒
        FETCH_RETRIES = 3  # 最多重试2次 (总共尝试3次)
        FETCH_RETRY_DELAY = 5 # 重试前等待5秒

        # --- 1. 入口处进行存活检查 ---
        if not apage or apage.is_closed():
            error_msg = "Page was already closed before processing observation."
            logger.warning(f"[async_process] {error_msg}")
            error_img = np.zeros((self.viewport_size["height"], self.viewport_size["width"], 3), dtype=np.uint8)
            return error_img, f"Error: {error_msg}"

        browser = apage.context.browser
        if not browser or not browser.is_connected():
            error_msg = "Browser was already closed before processing observation."
            logger.warning(f"[async_process] {error_msg}")
            error_img = np.zeros((self.viewport_size["height"], self.viewport_size["width"], 3), dtype=np.uint8)
            return error_img, f"Error: {error_msg}"


        # --- 2. 统一的、带超时的操作块 ---
        try:
            # --- 2a. 带有安全重试逻辑的 browser_info 获取 ---
            browser_info = None
            last_fetch_exception = None

            for attempt in range(FETCH_RETRIES + 1):
                try:
                    logger.debug(f"context_id {self.context_id} url {apage.url} [async_process] Attempt {attempt + 1} to fetch browser info...")
                    browser_info = await self.async_fetch_browser_info(apage)
                    logger.debug(f"context_id {self.context_id} url {apage.url} [async_process] {attempt + 1} async_fetch_browser_info finished")
                    # 如果成功，跳出重试循环
                    last_fetch_exception = None
                    break
                except Exception as e:
                    last_fetch_exception = e
                    logger.warning(f"[async_process] Attempt {attempt + 1} failed: {e}")
                    # 关键：检查是否是致命错误，如果是，则不应重试，立即向上抛出异常
                    if "closed" in str(e).lower():
                        raise e # 重新抛出致命错误，由外层 except 捕获

                    if attempt < FETCH_RETRIES:
                        logger.info(f"[async_process] Retrying in {FETCH_RETRY_DELAY} seconds...")
                        await asyncio.sleep(FETCH_RETRY_DELAY)

            # 如果所有重试都失败了，抛出最后的异常
            if last_fetch_exception:
                logger.error(f"context_id {self.context_id} url {apage.url} errot {last_fetch_exception}")
                raise last_fetch_exception

            # 检查获取到的 browser_info 是否有效
            if not browser_info or not browser_info.get("DOMTree"):
                error_msg = f"Failed to fetch DOM Tree after all retries: {browser_info.get('config', {}).get('error')}"
                logger.error(f"context_id {self.context_id} url {apage.url} error_msg {error_msg}")
                raise RuntimeError(error_msg)

            self.browser_config = browser_info["config"]

            # --- 2b. 后续的观察获取步骤 ---
            logger.debug(f"context_id {self.context_id} url {apage.url} errot get screenshot")
            screenshot_bytes = await apage.screenshot(timeout=DEFAULT_TIMEOUT)
            #logger.info(f"new_cdp_session context_id {self.context_id} url {apage.url} errot get screenshot by cdp")
            # cdp = await asyncio.wait_for(apage.context.new_cdp_session(apage), timeout=DEFAULT_TIMEOUT)
            # result = await asyncio.wait_for(cdp.send("Page.captureScreenshot", {"format": "png"}), timeout=DEFAULT_TIMEOUT)
            # screenshot_bytes = base64.b64decode(result["data"])

            screenshot_img = Image.open(BytesIO(screenshot_bytes))
            if self.observation_type == "image_som":
                som_bboxes = await self.async_get_page_bboxes(apage)
                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes, screenshot_img, viewport_size=self.viewport_size
                )
                # if VLM_EXP_DEBUG == '1':
                #     self.save_image(bbox_img, 'image_som')
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                return np.array(bbox_img), content_str
            else:
                # if VLM_EXP_DEBUG == '1':
                #     self.save_image(screenshot_img, 'image')
                if 'wPRSom'.lower() in EXPERIMENT_NAME.lower() or 'wPRALL'.lower() in EXPERIMENT_NAME.lower():
                    som_bboxes = await self.async_get_page_bboxes(apage)
                    bbox_img, id2center, content_str = self.draw_bounding_boxes(
                        som_bboxes, screenshot_img, viewport_size=self.viewport_size
                    )
                    self.som_id_info = id2center
                return np.array(screenshot_img), "success get the web screenshot_img"

        # --- 3. 统一的、信息明确的异常处理 ---
        except Exception as e:
            error_msg = f"Exception in async_process: {e}"
            logger.error(f"context_id {self.context_id} url {apage.url} [async_process] {error_msg}")
            error_img = np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3), dtype=np.uint8
            )
            return error_img, f"Error: {error_msg}"

    def get_element_center(self, element_id: str) -> tuple[float, float]:
        if not self.observation_type.startswith("image"):
            raise ValueError(f"get_element_center() is only supported for image observation types. {self.observation_type}")
        if isinstance(element_id, int):
            element_id = str(element_id)
        try:
            # Assumes som_id_info stores (center_x, center_y, width, height, is_interactable)
            center_x, center_y, _, _, is_interactable = self.som_id_info[element_id]
            if not is_interactable:
                 logger.warning(f"Element with id {element_id} is not marked as interactable.")
        except KeyError:
            raise ValueError(f"Cannot find element with id: {element_id} in the current viewport.")
        except (ValueError, IndexError):
             raise ValueError(f"Metadata for element id {element_id} is malformed.")

        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )

    def is_coords_on_interactable_element(self, x: float, y: float) -> bool:
        """
        Checks if the given (x, y) coordinates fall within the bounding box of any interactable element.

        Args:
            x (float): The x-coordinate, in absolute pixels.
            y (float): The y-coordinate, in absolute pixels.

        Returns:
            bool: True if the coordinates are on an interactable element, False otherwise.
        """
        # We iterate in reverse to check the elements that were drawn last (top-most) first.
        for element_data in reversed(list(self.som_id_info.values())):
            center_x, center_y, width, height, is_interactable = element_data

            # Calculate bounding box from center and width/height
            x0 = center_x - width / 2
            y0 = center_y - height / 2
            x1 = center_x + width / 2
            y1 = center_y + height / 2
            if VLM_EXP_DEBUG == '1':
                logger.debug(f"Coordinate check: x0={x0} x1={x1} y0={y0} y1={y1} | point x={x} y={y} | img_save={self.save_img_file}")

            if x0 <= x <= x1 and y0 <= y <= y1:
                # The first element we hit is the top-most one at these coordinates.
                # If it's interactable, our job is done.
                if is_interactable:
                    return True
                # If it's not interactable, no element below it can be interacted with at this coordinate.
                else:
                    return False

        # If the loop completes, the coordinate is not on any known element.
        return False
