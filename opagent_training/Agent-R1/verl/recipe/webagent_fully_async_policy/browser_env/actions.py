"""
Browser Env action space.
Inspited by Farama-Foundation/miniwob-plusplus
"""
import ast
import logging
import random
import re
import string
from enum import IntEnum
from itertools import chain
from pathlib import Path
from typing import Any, TypedDict, cast
from typing import Optional

import numpy as np
import numpy.typing as npt
from beartype import beartype
from beartype.door import is_bearable
from gymnasium import spaces
from playwright._impl._api_structures import ViewportSize
from playwright.async_api import BrowserContext as ABrowserContext
from playwright.async_api import Locator as ALocator
from playwright.async_api import Page as APage
from playwright.sync_api import BrowserContext, Locator, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from recipe.webagent_fully_async_policy.browser_env.processors import ObservationProcessor, AsyncImageObservationProcessor, ImageObservationProcessor
from recipe.webagent_fully_async_policy.browser_env.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    MAX_ANSWER_LENGTH,
    MAX_ELEMENT_ID,
    MAX_ELEMENT_INDEX_IN_VIEWPORT,
    MAX_PAGE_NUMBER,
    MAX_VANILLA_STR_LENGTH,
    PLAYWRIGHT_ACTIONS,
    PLAYWRIGHT_LOCATORS,
    ROLES,
    SPECIAL_KEY_MAPPINGS,
    SPECIAL_KEYS,
    SPECIAL_LOCATORS,
    TEXT_MAX_LENGTH,
    TYPING_MAX_LENGTH,
    URL_MAX_LENGTH,
    RolesType,
)
from recipe.webagent_fully_async_policy.browser_env.utils import with_timeout_legacy

import os
EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', '')

# 配置logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

domcontentloaded_timeout = 15 * 1000
ACTION_POST_WAIT_TIMEOUT = 15 * 1000  # 毫秒
class ParsedPlaywrightCode(TypedDict):
    function_name: str
    arguments: list[str]
    keywords: dict[str, Any]


from recipe.webagent_fully_async_policy.browser_env.processors import (
    ObservationProcessor,
)


@beartype
def is_in_viewport(
    element: Locator, viewport: ViewportSize, threshold: float = 0.3
) -> bool:
    """Given a playwright locator, check if it is in the viewport"""
    box = element.bounding_box()
    assert box is not None
    boxx0 = box["x"]
    boxx1 = box["x"] + box["width"]
    boxy0 = box["y"]
    boxy1 = box["y"] + box["height"]
    viewportx0, viewporty0 = 0, 0
    viewportx1, viewporty1 = viewport["width"], viewport["height"]
    inter = max(0, min(boxx1, viewportx1) - max(boxx0, viewportx0)) * max(
        0, min(boxy1, viewporty1) - max(boxy0, viewporty0)
    )
    ratio = inter / (box["width"] * box["height"])
    return ratio > threshold


@beartype
async def async_is_in_viewport(
    element: ALocator, viewport: ViewportSize, threshold: float = 0.3
) -> bool:
    box = await element.bounding_box()
    assert box is not None
    boxx0 = box["x"]
    boxx1 = box["x"] + box["width"]
    boxy0 = box["y"]
    boxy1 = box["y"] + box["height"]
    viewportx0, viewporty0 = 0, 0
    viewportx1, viewporty1 = viewport["width"], viewport["height"]
    inter = max(0, min(boxx1, viewportx1) - max(boxx0, viewportx0)) * max(
        0, min(boxy1, viewporty1) - max(boxy0, viewporty0)
    )
    ratio = inter / (box["width"] * box["height"])
    return ratio > threshold


class Action(TypedDict):
    action_type: int
    coords: npt.NDArray[np.float32]
    element_role: int
    element_name: str
    text: list[int]
    page_number: int
    url: str
    nth: int
    element_id: str
    direction: str
    key_comb: str
    pw_code: str
    answer: str
    raw_prediction: str  # raw prediction from the model
    press_enter_after: int


@beartype
def action2str(
    action: Action, action_set_tag: str, semantic_element: str = ""
) -> str:
    """Return the string representation of an action

    sementic_element: the semantic information of the element
    such as a line in an accessibility tree
    """
    if action_set_tag in [
        "id_accessibility_tree",
        "id_accessibility_tree_with_captioner",
    ]:
        element_id = action["element_id"]
        match action["action_type"]:
            case ActionTypes.CLICK:
                # [ID=X] xxxxx
                action_str = f"click [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.TYPE:
                text = "".join([_id2key[i] for i in action["text"]])
                action_str = f"type [{element_id}] [{text}] where [{element_id}] is {semantic_element}"
            case ActionTypes.HOVER:
                action_str = f"hover [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.SCROLL:
                action_str = f"scroll [{action['direction']}]"
            case ActionTypes.KEY_PRESS:
                action_str = f"press [{action['key_comb']}]"
            case ActionTypes.GOTO_URL:
                action_str = f"goto [{action['url']}]"
            case ActionTypes.NEW_TAB:
                action_str = "new_tab"
            case ActionTypes.PAGE_CLOSE:
                action_str = "close_tab"
            case ActionTypes.GO_BACK:
                action_str = "go_back"
            case ActionTypes.GO_FORWARD:
                action_str = "go_forward"
            case ActionTypes.PAGE_FOCUS:
                action_str = f"page_focus [{action['page_number']}]"
            case ActionTypes.CLEAR:
                action_str = f"clear [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.UPLOAD:
                action_str = f"upload [{action['text']}] to [{element_id}]"
            case ActionTypes.STOP:
                action_str = f"stop [{action['answer']}]"
            case ActionTypes.NONE:
                action_str = "none"
            case _:
                raise ValueError(
                    f"Unknown action type {action['action_type']}"
                )
    elif action_set_tag == "som":
        element_id = action["element_id"]
        match action["action_type"]:
            case ActionTypes.CLICK:
                # [ID=X] xxxxx
                action_str = f"click [{element_id}] where [{element_id}]"
            case ActionTypes.CLEAR:
                action_str = f"clear [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.TYPE:
                text = "".join([_id2key[i] for i in action["text"]])
                action_str = (
                    f"type [{element_id}] [{text}] where [{element_id}]"
                )
            case ActionTypes.HOVER:
                action_str = f"hover [{element_id}] where [{element_id}]"
            case ActionTypes.SCROLL:
                action_str = f"scroll [{action['direction']}]"
            case ActionTypes.KEY_PRESS:
                action_str = f"press [{action['key_comb']}]"
            case ActionTypes.GOTO_URL:
                action_str = f"goto [{action['url']}]"
            case ActionTypes.NEW_TAB:
                action_str = "new_tab"
            case ActionTypes.PAGE_CLOSE:
                action_str = "close_tab"
            case ActionTypes.GO_BACK:
                action_str = "go_back"
            case ActionTypes.GO_FORWARD:
                action_str = "go_forward"
            case ActionTypes.PAGE_FOCUS:
                action_str = f"page_focus [{action['page_number']}]"
            case ActionTypes.STOP:
                action_str = f"stop [{action['answer']}]"
            case ActionTypes.UPLOAD:
                action_str = f"upload [{action['text']}] to [{element_id}]"
            case ActionTypes.NONE:
                action_str = "none"
            case _:
                raise ValueError(
                    f"Unknown action type {action['action_type']}"
                )
    else:
        raise NotImplementedError(f"Unknown action set tag {action_set_tag}")

    return action_str


def action2create_function(action: Action) -> str:
    match (action["action_type"]):
        case ActionTypes.NONE:
            return "create_none_action()"
        # mouse wheel and keyboard action
        case ActionTypes.SCROLL:
            direction = "up" if "up" in action["direction"] else "down"
            return f"create_scroll_action({repr(direction)})"
        case ActionTypes.KEY_PRESS:
            return f"create_key_press_action({repr(action['key_comb'])})"
        # inter-page actions
        case ActionTypes.PAGE_FOCUS:
            return f"create_page_focus_action({action['page_number']})"
        case ActionTypes.NEW_TAB:
            return "create_new_tab_action()"
        case ActionTypes.GO_BACK:
            return "create_go_back_action()"
        case ActionTypes.GO_FORWARD:
            return "create_go_forward_action()"
        case ActionTypes.GOTO_URL:
            return f"create_goto_url_action({repr(action['url'])})"
        case ActionTypes.PAGE_CLOSE:
            return "create_page_close_action()"

        # low-level keyboard and mouse actions
        case ActionTypes.MOUSE_CLICK:
            return f"create_mouse_click_action({action['coords'][0]}, {action['coords'][1]})"
        case ActionTypes.MOUSE_HOVER:
            return f"create_mouse_hover_action({action['coords'][0]}, {action['coords'][1]})"
        case ActionTypes.KEYBOARD_TYPE:
            return f"create_keyboard_type_action({list(map(lambda x: _id2key[x], action['text']))})"

        # mid-level keyboard and mouse actions
        case ActionTypes.CLICK:
            args = []
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(
                f"element_role={repr(_id2role[action['element_role']])}"
            )
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_click_action({args_str})"
        case ActionTypes.CLEAR:
            args = []
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(
                f"element_role={repr(_id2role[action['element_role']])}"
            )
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_clear_action({args_str})"
        case ActionTypes.UPLOAD:
            args = []
            text = "".join(map(lambda x: _id2key[x], action["text"]))
            args.append(f"text={repr(text)}")
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(
                f"element_role={repr(_id2role[action['element_role']])}"
            )
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_upload_action({args_str})"
        case ActionTypes.HOVER:
            args = []
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(
                f"element_role={repr(_id2role[action['element_role']])}"
            )
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_hover_action({args_str})"
        case ActionTypes.TYPE:
            args = []
            text = "".join(map(lambda x: _id2key[x], action["text"]))
            args.append(f"text={repr(text)}")
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(
                f"element_role={repr(_id2role[action['element_role']])}"
            )
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_type_action({args_str})"

        # high-level actions, only support locators from playwright
        case ActionTypes.CHECK:
            return f"create_check_action(pw_code={repr(action['pw_code'])})"
        case ActionTypes.SELECT_OPTION:
            return f"create_select_option_action(pw_code={repr(action['pw_code'])})"
        case ActionTypes.STOP:
            return f'create_stop_action({repr(action["answer"])})'
        
        # New coordinate-based actions
        case ActionTypes.DOUBLE_CLICK:
            return f"create_mouse_click_action({action['coords'][0]}, {action['coords'][1]})"  # Note: may need a dedicated create function
        case ActionTypes.MOVE_TO:
            return f"create_mouse_hover_action({action['coords'][0]}, {action['coords'][1]})"
        case ActionTypes.COORDS_SELECT_OPTION:
            return f"create_mouse_click_action({action['coords'][0]}, {action['coords'][1]})"  # Simplified
        case ActionTypes.WAIT:
            return f"create_none_action()"  # Note: may need a dedicated create function
        case ActionTypes.FIND:
            occurrence = action.get('occurrence', 0)
            highlight = action.get('highlight', True)
            return f"create_find_action({repr(action['search_term'])}, {occurrence}, {highlight})"
        case ActionTypes.GET_FILE:
            return f"create_get_file_action({action['coords'][0]}, {action['coords'][1]})"

    raise ValueError(f"Invalid action type: {action['action_type']}")


class ActionTypes(IntEnum):
    """Valid action types for browser env."""

    NONE = 0
    # mouse wheel and keyboard, universal across all action spaces
    SCROLL = 1
    KEY_PRESS = 2

    # low level mouse and keyboard actions
    MOUSE_CLICK = 3
    KEYBOARD_TYPE = 4
    MOUSE_HOVER = 5

    # mid level mouse and keyboard actions
    CLICK = 6  # ID/Semantic-based click
    TYPE = 7   # ID/Semantic-based type
    HOVER = 8  # ID/Semantic-based hover

    # page level actions, universal across all action spaces
    PAGE_FOCUS = 9
    NEW_TAB = 10
    GO_BACK = 11
    GO_FORWARD = 12
    GOTO_URL = 13
    PAGE_CLOSE = 14

    # high-leval actions that playwright support
    CHECK = 15
    SELECT_OPTION = 16

    STOP = 17
    CLEAR = 18
    UPLOAD = 19
    
    # --- New Coordinate-Based Actions ---
    
    # Double click at a specific coordinate. MOUSE_CLICK is for single click.
    DOUBLE_CLICK = 20
    
    # Explicitly move the mouse to a coordinate without clicking.
    # While MOUSE_HOVER can be used, MOVE_TO is more semantically clear
    # and matches the JSON schema. We can map them to the same execution logic.
    MOVE_TO = 21

    # Select an option from a dropdown located at a specific coordinate.
    # SELECT_OPTION is for Playwright-selector-based selection.
    COORDS_SELECT_OPTION = 22
    
    # A dedicated wait action.
    WAIT = 23
    
    # Search for text on the page
    FIND = 24
    
    # Download file or get image from a specific coordinate
    GET_FILE = 25

    def __str__(self) -> str:
        return f"ACTION_TYPES.{self.name}"


@beartype
def is_equivalent(a: Action, b: Action) -> bool:
    """Return True if two actions are equal."""
    if a["action_type"] != b["action_type"]:
        return False
    match (a["action_type"]):
        case ActionTypes.NONE:
            return True
        case ActionTypes.SCROLL:
            return a["direction"] == b["direction"]
        case ActionTypes.KEY_PRESS:
            return a["key_comb"] == b["key_comb"]
        case ActionTypes.MOUSE_CLICK | ActionTypes.MOUSE_HOVER:
            return np.allclose(a["coords"], b["coords"])
        case ActionTypes.KEYBOARD_TYPE:
            return a["text"] == b["text"]
        case ActionTypes.CLICK | ActionTypes.HOVER | ActionTypes.TYPE:  # TODO: can be further optimized
            if a["element_id"] and b["element_id"]:
                return a["element_id"] == b["element_id"]
            elif a["element_role"] and b["element_role"]:
                return (
                    a["element_role"] == b["element_role"]
                    and a["element_name"] == b["element_name"]
                )
            elif a["pw_code"] and b["pw_code"]:
                return a["pw_code"] == b["pw_code"]
            else:
                return False
        case ActionTypes.PAGE_FOCUS:
            return a["page_number"] == b["page_number"]
        case ActionTypes.NEW_TAB:
            return True
        case ActionTypes.GO_BACK:
            return True
        case ActionTypes.GO_FORWARD:
            return True
        case ActionTypes.GOTO_URL:
            return a["url"] == b["url"]
        case ActionTypes.PAGE_CLOSE:
            return True
        case ActionTypes.CHECK | ActionTypes.SELECT_OPTION:
            return a["pw_code"] == b["pw_code"]
        case ActionTypes.STOP:
            return a["answer"] == b["answer"]
        case ActionTypes.FIND:
            return a["search_term"] == b["search_term"] and a.get("occurrence", 0) == b.get("occurrence", 0)
        case ActionTypes.GET_FILE:
            return np.allclose(a["coords"], b["coords"])
        case ActionTypes.DOUBLE_CLICK | ActionTypes.MOVE_TO:
            return np.allclose(a["coords"], b["coords"])
        case ActionTypes.COORDS_SELECT_OPTION:
            return np.allclose(a["coords"], b["coords"]) and a["element_name"] == b["element_name"]
        case ActionTypes.WAIT:
            return a["nth"] == b["nth"]
        case _:
            raise ValueError(f"Unknown action type: {a['action_type']}")


_key2id: dict[str, int] = {
    key: i
    for i, key in enumerate(
        chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"])
    )
}
_id2key: list[str] = sorted(_key2id, key=_key2id.get)  # type: ignore[arg-type]
_role2id: dict[RolesType, int] = {
    cast(RolesType, role): i
    for i, role in enumerate(chain(ROLES, SPECIAL_LOCATORS))
}
_id2role: list[RolesType] = sorted(_role2id, key=_role2id.get)  # type: ignore[arg-type]


@beartype
def _keys2ids(keys: list[int | str] | str) -> list[int]:
    return list(
        map(
            lambda key: _key2id.get(str(key), _key2id.get(key, _key2id.get(" ", 0)))
            if is_bearable(key, str)
            else int(key),
            keys,
        )
    )


def get_action_space() -> spaces.Dict:
    """Return the space of serialized actions."""
    space = spaces.Dict(
        {
            "action_type": spaces.Discrete(len(ActionTypes)),
            # coords (left, top) is used for COORD_CLICK
            "coords": spaces.Box(
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
            ),
            # element role is used for FOCUS_AND_CLICK and FOCUS_AND_TYPE
            "element_role": spaces.Discrete(
                len(ROLES) + len(SPECIAL_LOCATORS)
            ),
            # element name is used with element role
            "element_name": spaces.Text(TEXT_MAX_LENGTH),
            "element_id": spaces.Text(TEXT_MAX_LENGTH),
            # text is only used for TYPE and FOCUS_AND_TYPE
            "text": spaces.MultiDiscrete(
                [
                    len(ASCII_CHARSET)
                    + len(SPECIAL_KEYS)
                    + len(FREQ_UNICODE_CHARSET)
                ]
                * TYPING_MAX_LENGTH
            ),
            "page_number": spaces.Discrete(MAX_PAGE_NUMBER),
            "url": spaces.Text(URL_MAX_LENGTH),
            "nth": spaces.Discrete(MAX_ELEMENT_INDEX_IN_VIEWPORT),
            "key_comb": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "direction": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "pw_code": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "answer": spaces.Text(MAX_ANSWER_LENGTH),
        }
    )
    return space


def create_random_action() -> Action:
    """Return a random action."""
    return {
        "action_type": np.random.randint(len(ActionTypes)),
        "coords": np.random.rand(2).astype(np.float32),
        "element_role": np.random.randint(len(ROLES) + len(SPECIAL_LOCATORS)),
        "element_name": "".join(
            random.choices(ASCII_CHARSET, k=np.random.randint(TEXT_MAX_LENGTH))
        ),
        "text": list(
            random.choices(
                list(range(len(ASCII_CHARSET))),
                k=np.random.randint(TYPING_MAX_LENGTH),
            )
        ),
        "page_number": np.random.randint(MAX_PAGE_NUMBER),
        "url": "".join(
            random.choices(ASCII_CHARSET, k=np.random.randint(URL_MAX_LENGTH))
        ),
        "nth": np.random.randint(MAX_ELEMENT_INDEX_IN_VIEWPORT),
        "element_id": str(np.random.randint(MAX_ELEMENT_ID)),
        "key_comb": "+".join(
            random.choices(SPECIAL_KEYS, k=np.random.randint(3))
        ),
        "direction": random.choice(["up", "down"]),
        "pw_code": "".join(
            random.choices(
                string.ascii_uppercase + string.digits,
                k=np.random.randint(MAX_VANILLA_STR_LENGTH),
            )
        ),
        "answer": str(np.random.randint(MAX_ANSWER_LENGTH)),
        "raw_prediction": str(np.random.randint(MAX_ANSWER_LENGTH)),
    }


@beartype
def create_none_action() -> Action:
    """Return a valid action object that does nothing."""
    return {
        "action_type": ActionTypes.NONE,
        "coords": np.zeros(2, dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "pw_code": "",  # str that requires further processing
        "element_id": "",
        "key_comb": "",
        "direction": "",
        "answer": "",
        "raw_prediction": "",
    }

@beartype
def create_stop_action(answer: str) -> Action:
    action = create_none_action()
    action.update({"action_type": ActionTypes.STOP, "answer": answer})
    return action


@beartype
def create_scroll_action(direction: str) -> Action:
    """Return the playwright action"""
    assert direction in ["up", "down"]
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.SCROLL,
            "direction": direction,
        }
    )
    return action


@beartype
def create_mouse_hover_action(
    left: float | None = None, top: float | None = None
) -> Action:
    """Return a valid action object with type COORD_CLICK."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.MOUSE_HOVER,
            "coords": np.array([left, top], dtype=np.float32),
        }
    )
    return action


@beartype
def create_key_press_action(key_comb: str) -> Action:
    """Return the key press action"""

    def map_keys(key_comb: str) -> str:
        keys = key_comb.split("+")
        mapped_keys = []
        for key in keys:
            mapped_key = SPECIAL_KEY_MAPPINGS.get(key.lower(), key)
            mapped_keys.append(mapped_key)
        return "+".join(mapped_keys)

    action = create_none_action()
    mapped_key_comb = map_keys(key_comb)
    action.update(
        {
            "action_type": ActionTypes.KEY_PRESS,
            "key_comb": mapped_key_comb,
        }
    )
    return action


@beartype
def create_page_focus_action(page_number: int) -> Action:
    """Return a valid action object with type PAGE_FOCUS."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.PAGE_FOCUS,
            "page_number": page_number,
        }
    )
    return action


@beartype
def create_new_tab_action() -> Action:
    """Return a valid action object with type NEW_TAB."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.NEW_TAB,
        }
    )
    return action


@beartype
def create_go_back_action() -> Action:
    """Return a valid action object with type GO_BACK."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.GO_BACK,
        }
    )
    return action


@beartype
def create_go_forward_action() -> Action:
    """Return a valid action object with type GO_FORWARD."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.GO_FORWARD,
        }
    )
    return action


@beartype
def create_goto_url_action(url: str) -> Action:
    """Return a valid action object with type GOTO_URL."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.GOTO_URL,
            "url": url,
        }
    )
    return action


@beartype
def create_page_close_action() -> Action:
    """Return a valid action object with type PAGE_CLOSE."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.PAGE_CLOSE,
        }
    )
    return action


@beartype
def create_find_action(search_term: str, occurrence: int = 0, highlight: bool = True) -> Action:
    """Return a valid action object with type FIND."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.FIND,
            "search_term": search_term,
            "occurrence": occurrence,
            "highlight": highlight,
        }
    )
    return action


@beartype
def create_get_file_action(left: float, top: float) -> Action:
    """Return a valid action object with type GET_FILE."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.GET_FILE,
            "coords": np.array([left, top], dtype=np.float32),
        }
    )
    return action


@beartype
def create_mouse_click_action(
    left: float | None = None, top: float | None = None
) -> Action:
    """Return a valid action object with type COORD_CLICK."""
    action = create_none_action()
    if left and top:
        action.update(
            {
                "action_type": ActionTypes.MOUSE_CLICK,
                "coords": np.array([left, top], dtype=np.float32),
            }
        )
    elif (not left) and (not top):
        action.update(
            {
                "action_type": ActionTypes.CLICK,
            }
        )
    else:
        raise ValueError("left and top must be both None or both not None")
    return action


@beartype
def create_clear_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CLEAR,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "pw_code": pw_code,
        }
    )
    return action

@beartype
def create_upload_action(
    text: str,
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.TYPE,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "text": _keys2ids(text),
            "pw_code": pw_code,
        }
    )
    return action

@beartype
def create_keyboard_type_action(keys: list[int | str] | str) -> Action:
    """Return a valid action object with type TYPE."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.KEYBOARD_TYPE,
            "text": _keys2ids(keys),
        }
    )
    return action


@beartype
def create_click_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CLICK,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_hover_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.HOVER,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_type_action(
    text: str,
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.TYPE,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "text": _keys2ids(text),
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_check_action(pw_code: str) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CHECK,
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_select_option_action(
    pw_code: str,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.SELECT_OPTION,
            "pw_code": pw_code,
        }
    )
    return action


@beartype
def create_focus_action(
    element_role: RolesType, element_name: str = "", nth: int = 0
) -> Action:
    """Return a valid action object with type CLICK.

    Keep compatible with the old version."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CLICK,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
        }
    )
    return action


@beartype
def create_focus_and_click_action(
    element_role: RolesType, element_name: str = "", nth: int = 0
) -> Action:
    """Return a valid action object with type CLICK.

    Keep compatible with the old version."""

    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CLICK,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
        }
    )
    return action


@beartype
def create_focus_and_type_action(
    keys: list[int | str] | str,
    element_role: RolesType,
    element_name: str = "",
    nth: int = 0,
) -> Action:
    """Return a valid action object with type TYPE.

    Keep compatible with the old version."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.TYPE,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "text": _keys2ids(keys),
            "nth": nth,
        }
    )
    return action


@beartype
def execute_scroll(direction: str, page: Page) -> None:
    # perform the action
    # code from natbot
    if direction == "up":
        page.evaluate(
            "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
        )
    elif direction == "down":
        page.evaluate(
            "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
        )


@beartype
async def aexecute_scroll(direction: str, page: APage) -> None:
    # perform the action
    # code from natbot
    if direction == "up":
        await page.evaluate(
            "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
        )
    elif direction == "down":
        await page.evaluate(
            "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
        )


@beartype
def execute_key_press(key: str, page: Page) -> None:
    """Press a key."""
    if "Meta" in key and "Mac" not in page.evaluate("navigator.platform"):
        key = key.replace("Meta", "Control")
    page.keyboard.press(key)


@beartype
async def aexecute_key_press(key: str, page: APage) -> None:
    """Press a key."""
    if "Meta" in key and "Mac" not in await page.evaluate(
        "navigator.platform"
    ):
        key = key.replace("Meta", "Control")
    await page.keyboard.press(key)


@beartype
def execute_mouse_hover(left: float, top: float, page: Page) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    page.mouse.move(
        left * viewport_size["width"], top * viewport_size["height"]
    )


@beartype
async def aexecute_mouse_hover(left: float, top: float, page: APage) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    await page.mouse.move(
        left * viewport_size["width"], top * viewport_size["height"]
    )


def execute_mouse_click(left: float, top: float, page: Page) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    page.mouse.click(
        left * viewport_size["width"], top * viewport_size["height"]
    )


@beartype
async def aexecute_mouse_click(left: float, top: float, page: APage, click_count: int = 1) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    await page.mouse.click(
        left * viewport_size["width"], top * viewport_size["height"], click_count=click_count
    )


def execute_upload(left: float, top: float, path: str, page: Page) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    with page.expect_file_chooser() as fc_info:
        page.mouse.click(
            left * viewport_size["width"], top * viewport_size["height"]
        )
    file_chooser = fc_info.value
    file_chooser.set_files(path)


@beartype
async def aexecute_upload(left: float, top: float, path: str, page: APage) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    async with page.expect_file_chooser() as fc_info:
        await page.mouse.click(
            left * viewport_size["width"], top * viewport_size["height"]
        )
    file_chooser = fc_info.value
    await file_chooser.set_files(path)



@beartype
def execute_keyboard_type(text: str, page: Page) -> None:
    """Fill the focused element with text."""
    page.keyboard.type(text)


@beartype
async def aexecute_keyboard_type(text: str, page: APage) -> None:
    """Fill the focused element with text."""
    await page.keyboard.type(text)


@beartype
def execute_click_current(page: Page) -> None:
    """Click at the current mouse position."""
    locators = page.locator("*:focus")
    if not locators.count():
        for frame in page.frames[1:]:
            locators = frame.locator("*:focus")
            if locators.count():
                break
    locators.click()


@beartype
async def aexecute_click_current(page: APage) -> None:
    """Click at the current mouse position."""
    locators = page.locator("*:focus")
    locator_count = await locators.count()
    if not locator_count:
        for frame in page.frames[1:]:
            locators = frame.locator("*:focus")
            locator_count = await locators.count()
            if locator_count:
                break
    await locators.click()
    await page.wait_for_load_state("load", timeout=60 * 1000)


@beartype
def execute_type(keys: list[int], page: Page) -> None:
    """Send keystrokes to the focused element."""
    text = "".join([_id2key[key] for key in keys])
    page.keyboard.type(text)


@beartype
async def aexecute_type(keys: list[int], page: APage) -> None:
    """Send keystrokes to the focused element."""
    text = "".join([_id2key[key] for key in keys])
    await page.keyboard.type(text)


@beartype
def execute_focus(
    element_role: int, element_name: str, nth: int, page: Page
) -> None:
    """Click the specified DOM element."""
    element_role_str = _id2role[element_role]
    if page.viewport_size is None:
        raise ValueError("Viewport size is not set for the current page")
    element_location_list: list[tuple[Locator, float, float]] = []
    for frame in page.frames:
        match element_role_str:
            case "alt_text":
                locators = frame.get_by_alt_text(element_name)
            case "label":
                locators = frame.get_by_label(element_name)
            case "placeholder":
                locators = frame.get_by_placeholder(element_name)
            case _:
                locators = frame.get_by_role(
                    role=element_role_str, name=element_name
                )
        for locator_idx in range(locators.count()):
            locator = locators.nth(locator_idx)
            if is_in_viewport(locator, page.viewport_size):
                bounding_box = locator.bounding_box()
                assert bounding_box
                element_location_list.append(
                    (locator, bounding_box["x"], bounding_box["y"])
                )
    if len(element_location_list) <= nth:
        raise ValueError(
            f"There are only {len(element_location_list)} elements found in viewport, but {nth + 1} is requested"
        )
    element_location_list.sort(key=lambda x: (x[2], x[1]))  # row major order
    element_location_list[nth][0].focus()


@beartype
async def aexecute_focus(
    element_role: int, element_name: str, nth: int, page: APage
) -> None:
    """Click the specified DOM element."""
    element_role_str = _id2role[element_role]
    if page.viewport_size is None:
        raise ValueError("Viewport size is not set for the current page")
    element_location_list: list[tuple[ALocator, float, float]] = []
    for frame in page.frames:
        match element_role_str:
            case "alt_text":
                locators = frame.get_by_alt_text(element_name)
            case "label":
                locators = frame.get_by_label(element_name)
            case "placeholder":
                locators = frame.get_by_placeholder(element_name)
            case _:
                locators = frame.get_by_role(
                    role=element_role_str, name=element_name
                )
        for locator_idx in range(await locators.count()):
            locator = locators.nth(locator_idx)
            if await async_is_in_viewport(locator, page.viewport_size):
                bounding_box = await locator.bounding_box()
                assert bounding_box
                element_location_list.append(
                    (locator, bounding_box["x"], bounding_box["y"])
                )
    if len(element_location_list) <= nth:
        raise ValueError(
            f"There are only {len(element_location_list)} elements found in viewport, but {nth + 1} is requested"
        )
    element_location_list.sort(key=lambda x: (x[2], x[1]))  # row major order
    await element_location_list[nth][0].focus()


@beartype
def locate(locator_calls: list[ParsedPlaywrightCode], page: Page) -> Locator:
    locator = page
    for call in locator_calls:
        function_name = call["function_name"]
        arguments = call["arguments"]
        keywords = call["keywords"]
        locator = getattr(locator, function_name)(*arguments, **keywords)
    return locator  # type: ignore[return-value]


@beartype
async def alocate(
    locator_calls: list[ParsedPlaywrightCode], page: APage
) -> ALocator:
    locator = page
    for call in locator_calls:
        function_name = call["function_name"]
        arguments = call["arguments"]
        keywords = call["keywords"]
        locator = await getattr(locator, function_name)(*arguments, **keywords)
    return locator  # type: ignore[return-value]


@beartype
def execute_playwright_click(
    locator_code: list[ParsedPlaywrightCode],
    page: Page,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)

    # perform the action
    locator.click(*pw_action_args, **pw_action_kwargs)


@beartype
async def aexecute_playwright_click(
    locator_code: list[ParsedPlaywrightCode],
    page: APage,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = await alocate(locator_code, page)

    # perform the action
    await locator.click(*pw_action_args, **pw_action_kwargs)


@beartype
def execute_playwright_hover(
    locator_code: list[ParsedPlaywrightCode], page: Page
) -> None:
    locator = locate(locator_code, page)

    # perform the action
    locator.hover()


@beartype
async def aexecute_playwright_hover(
    locator_code: list[ParsedPlaywrightCode], page: APage
) -> None:
    locator = await alocate(locator_code, page)

    # perform the action
    await locator.hover()


@beartype
def execute_playwright_type(
    text: str,
    locator_code: list[ParsedPlaywrightCode],
    page: Page,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)
    # perform the action
    pw_action_args = [text] + pw_action_args  # text is the first argument
    locator.type(*pw_action_args, **pw_action_kwargs)


@beartype
async def aexecute_playwright_type(
    text: str,
    locator_code: list[ParsedPlaywrightCode],
    page: APage,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = await alocate(locator_code, page)
    # perform the action
    pw_action_args = [text] + pw_action_args  # text is the first argument
    await locator.type(*pw_action_args, **pw_action_kwargs)


@beartype
def execute_playwright_select_option(
    locator_code: list[ParsedPlaywrightCode],
    page: Page,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)
    # perform the action
    locator.select_option(*pw_action_args, **pw_action_kwargs)


@beartype
async def aexecute_playwright_select_option(
    locator_code: list[ParsedPlaywrightCode],
    page: APage,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = await alocate(locator_code, page)
    # perform the action
    await locator.select_option(*pw_action_args, **pw_action_kwargs)


@beartype
def execute_playwright_check(
    locator_code: list[ParsedPlaywrightCode], page: Page
) -> None:
    locator = locate(locator_code, page)
    # perform the action
    locator.check()


@beartype
async def aexecute_playwright_check(
    locator_code: list[ParsedPlaywrightCode], page: APage
) -> None:
    locator = await alocate(locator_code, page)
    # perform the action
    await locator.check()


@beartype
def execute_action(
    action: Action,
    page: Page,
    browser_ctx: BrowserContext,
    obseration_processor: ObservationProcessor,
    sleep_after_execution: float = 0.0, # 这个参数现在可以被更智能的等待取代
) -> Page:
    """Execute the action on the ChromeDriver."""
    action_type = action["action_type"]
    num_tabs_before = len(browser_ctx.pages)
    
    # --- 原始的 match action_type 逻辑保持完全不变 ---
    # ... (从 match action_type: 到 case _: 的所有代码都保持原样)
    match action_type:
        case ActionTypes.NONE:
            pass

        case ActionTypes.SCROLL:
            direction = "up" if "up" in action["direction"] else "down"
            execute_scroll(direction, page)
        case ActionTypes.KEY_PRESS:
            keys = action["key_comb"]
            execute_key_press(keys, page)

        case ActionTypes.MOUSE_CLICK:
            execute_mouse_click(action["coords"][0], action["coords"][1], page)
        case ActionTypes.CLEAR:
            element_id = action["element_id"]
            element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            execute_mouse_click(element_center[0], element_center[1], page)
            execute_key_press("Meta+A", page)
            execute_key_press('Backspace', page)
        case ActionTypes.MOUSE_HOVER:
            execute_mouse_hover(action["coords"][0], action["coords"][1], page)
        case ActionTypes.KEYBOARD_TYPE:
            execute_type(action["text"], page)
        case ActionTypes.CLICK:
            # check each kind of locator in order
            # TODO[shuyanzh]: order is temp now
            if action["element_id"]:
                element_id = action["element_id"]
                element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                execute_mouse_click(element_center[0], element_center[1], page)
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                execute_focus(element_role, element_name, nth, page)
                execute_click_current(page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                # [shuyanzh], don't support action args and kwargs now
                execute_playwright_click(locator_code=locator_code, page=page)
            else:
                raise ValueError("No proper locator found for click action")
        case ActionTypes.HOVER:
            if action["element_id"]:
                element_id = action["element_id"]
                element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                execute_mouse_hover(element_center[0], element_center[1], page)
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                execute_focus(element_role, element_name, nth, page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                # [shuyanzh], don't support action args and kwargs now
                execute_playwright_hover(locator_code=locator_code, page=page)
            else:
                raise NotImplementedError(
                    "No proper locator found for hover action"
                )
        case ActionTypes.TYPE:
            if action["element_id"]:
                element_id = action["element_id"]
                element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                execute_mouse_click(element_center[0], element_center[1], page)
                execute_type(action["text"], page)
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                execute_focus(element_role, element_name, nth, page)
                execute_type(action["text"], page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                text = parsed_code[-1]["arguments"][0]
                # [shuyanzh], don't support action args and kwargs now
                execute_playwright_type(
                    text=text, locator_code=locator_code, page=page
                )
            else:
                raise NotImplementedError(
                    "No proper locator found for type action"
                )

        case ActionTypes.PAGE_FOCUS:
            page = browser_ctx.pages[int(action["page_number"])]
            page.bring_to_front()
        case ActionTypes.NEW_TAB:
            page = browser_ctx.new_page()
            page.goto("https://www.baidu.com", wait_until="networkidle")

        case ActionTypes.GO_BACK:
            if len(browser_ctx.pages) > 1:
                page.go_back()
        case ActionTypes.GO_FORWARD:
            page.go_forward()
        case ActionTypes.GOTO_URL:
            page.goto(action["url"])
        case ActionTypes.PAGE_CLOSE:
            if len(browser_ctx.pages) > 1:
                page.close()
                if len(browser_ctx.pages) > 0:
                    page = browser_ctx.pages[-1]
                else:
                    page = browser_ctx.new_page()
                    page.goto("https://www.baidu.com", wait_until="networkidle")


        case ActionTypes.SELECT_OPTION:
            if action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                execute_playwright_select_option(locator_code, page)
            else:
                raise NotImplementedError(
                    "No proper locator found for select option action"
                )
        case ActionTypes.CHECK:
            if action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                execute_playwright_check(locator_code, page)
            else:
                raise NotImplementedError(
                    "No proper locator found for select option action"
                )
        case ActionTypes.UPLOAD:
            element_id = action["element_id"]
            element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            execute_upload(element_center[0], element_center[1], action["text"], page)
        case _:
            raise ValueError(f"Unknown action type: {action_type}")
    # --- 原始的 match 逻辑结束 ---


    # --- 新增和修改的逻辑 ---
    
    # 检查是否有新标签页被打开
    # 检查是否有新标签页被打开（例如，点击链接后在新窗口打开）
    # 1. 处理可能新打开的标签页
    #if len(browser_ctx.pages) > num_tabs_before:
    page = browser_ctx.pages[-1]
    #logger.info(f"aexecute_action_coords: 新打开的标签页: {page.url}")
    page.bring_to_front()

    # 关键修改：在任何可能导致导航的操作后，等待页面稳定。
    # 使用 try...except 来优雅地处理超时情况。
    try:
        # 这个调用会等待，直到新页面加载完成或超时。
        page.wait_for_load_state("networkidle", timeout=domcontentloaded_timeout)
    except Exception:
        # 如果没有发生导航，这个等待会超时。我们捕获这个异常并忽略它，
        # 因为这意味着页面是稳定的，我们可以安全地继续。
        pass

    return page

@with_timeout_legacy(2 * 60)
async def aexecute_action(
    action: Action,
    page: APage,
    browser_ctx: ABrowserContext,
    obseration_processor: AsyncImageObservationProcessor,
    # sleep_after_execution: float = 0.0, # 这个参数不再需要，因为我们有了更智能的等待
) -> APage:
    """Execute the async action on the ChromeDriver."""
    num_tabs_before = len(browser_ctx.pages)
    action_type = action["action_type"]

    # --- 原始的 match action_type 逻辑保持完全不变 ---
    # ... (从 match action_type: 到 case _: 的所有代码都保持原样)
    match action_type:
        case ActionTypes.NONE:
            pass
        case ActionTypes.SCROLL:
            direction = "up" if "up" in action["direction"] else "down"
            await aexecute_scroll(direction, page)
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.KEY_PRESS:
            keys = action["key_comb"]
            await aexecute_key_press(keys, page)
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.MOUSE_CLICK:
            await aexecute_mouse_click(
                action["coords"][0], action["coords"][1], page
            )
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.CLEAR:
            element_id = action["element_id"]
            element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            await aexecute_mouse_click(element_center[0], element_center[1], page)
            await page.wait_for_timeout(500)
            await aexecute_key_press("Meta+A", page)
            await aexecute_key_press('Backspace', page)
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.MOUSE_HOVER:
            await aexecute_mouse_hover(
                action["coords"][0], action["coords"][1], page
            )
            await page.wait_for_timeout(500)
        case ActionTypes.KEYBOARD_TYPE:
            await aexecute_type(action["text"], page)
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.CLICK:
            # check each kind of locator in order
            # TODO[shuyanzh]: order is temp now
            if action["element_id"]:
                element_id = action["element_id"]
                element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                await aexecute_mouse_click(element_center[0], element_center[1], page)
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                await aexecute_focus(element_role, element_name, nth, page)
                await aexecute_click_current(page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                # [shuyanzh], don't support action args and kwargs now
                await aexecute_playwright_click(
                    locator_code=locator_code, page=page
                )
            else:
                raise ValueError("No proper locator found for click action")
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.HOVER:
            if action["element_id"]:
                element_id = action["element_id"]
                element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                await aexecute_mouse_hover(element_center[0], element_center[1], page)
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                await aexecute_focus(element_role, element_name, nth, page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                # [shuyanzh], don't support action args and kwargs now
                await aexecute_playwright_hover(
                    locator_code=locator_code, page=page
                )
            else:
                raise NotImplementedError(
                    "No proper locator found for hover action"
                )
            await page.wait_for_timeout(500)
        case ActionTypes.TYPE:
            if action["element_id"]:
                element_id = action["element_id"]
                element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                #连点三次模拟全选
                await aexecute_mouse_click(element_center[0], element_center[1], page, click_count=3)
                await page.wait_for_timeout(500)
                await aexecute_key_press("Backspace", page)
                await page.wait_for_timeout(500)
                await aexecute_type(action["text"], page)
                await page.wait_for_timeout(500)
                await aexecute_key_press("Enter", page)
                await page.wait_for_timeout(2 * 1000)
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                await aexecute_focus(element_role, element_name, nth, page)
                await aexecute_type(action["text"], page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                text = parsed_code[-1]["arguments"][0]
                # [shuyanzh], don't support action args and kwargs now
                await aexecute_playwright_type(
                    text=text, locator_code=locator_code, page=page
                )
            else:
                raise NotImplementedError(
                    "No proper locator found for type action"
                )
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.PAGE_FOCUS:
            page = browser_ctx.pages[action["page_number"]]
            await page.bring_to_front()
        case ActionTypes.NEW_TAB:
            page = await browser_ctx.new_page()
            await page.goto("https://www.baidu.com", wait_until="networkidle")
        case ActionTypes.GO_BACK:
            if len(browser_ctx.pages) > 1:
                await page.go_back()
        case ActionTypes.GO_FORWARD:
            await page.go_forward()
        case ActionTypes.GOTO_URL:
            await page.goto(action["url"], wait_until="networkidle")
        case ActionTypes.PAGE_CLOSE:
            if len(browser_ctx.pages) > 1:
                await page.close()
                if len(browser_ctx.pages) > 0:
                    page = browser_ctx.pages[-1]
                else:
                    page = await browser_ctx.new_page()
                    await page.goto("https://www.baidu.com", wait_until="networkidle")
        case ActionTypes.SELECT_OPTION:
            if action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                await aexecute_playwright_select_option(locator_code, page)
            else:
                raise NotImplementedError(
                    "No proper locator found for select option action"
                )
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.CHECK:
            if action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                await aexecute_playwright_check(locator_code, page)
            else:
                raise NotImplementedError(
                    "No proper locator found for select option action"
                )
        case ActionTypes.UPLOAD:
            element_id = action["element_id"]
            element_center = obseration_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            await aexecute_upload(element_center[0], element_center[1], action["text"], page)        
        case _:
            raise ValueError(f"Unknown action type: {action_type}")
    # --- 原始的 match 逻辑结束 ---


    # --- 新增和修改的逻辑 ---
    
    # 检查是否有新标签页被打开（例如，点击链接后在新窗口打开）
    # 1. 处理可能新打开的标签页
    if len(browser_ctx.pages) > num_tabs_before:
        page = browser_ctx.pages[-1]
        #logger.info(f"aexecute_action_coords: 新打开的标签页: {page.url}")
        await page.bring_to_front()

    # 2. 在动作执行后等待页面稳定
    # 像点击、提交这样的动作可能会触发页面跳转。我们必须等待新页面准备就绪才能继续。
    try:
        # 'domcontentloaded' 是速度和就绪状态之间的一个很好的平衡点。
        # 它在初始HTML文档被完全加载和解析后触发，无需等待样式表、图片和子框架完成加载。
        await page.wait_for_load_state(
            "networkidle", timeout=ACTION_POST_WAIT_TIMEOUT
        )
    except PlaywrightTimeoutError:
        # 这不是一个错误。它仅仅意味着动作没有触发导航（例如，点击了一个非链接元素，在一个输入框里打字）。
        # 我们可以安全地忽略这个超时并继续执行。
        pass
    except Exception as e:
        # 动作执行期间，页面或上下文可能被关闭了。
        # 检查页面是否仍然可用。
        if page.is_closed():
            logger.warning(f"页面在执行 '{action_type}' 动作后被关闭。重新分配到一个有效的页面。")
            if len(browser_ctx.pages) > 0:
                page = browser_ctx.pages[-1]
            else:
                # 如果没有页面了，创建一个新的以避免崩溃。
                page = await browser_ctx.new_page()
                await page.goto("https://www.baidu.com", wait_until="networkidle")
        else:
            # 重新抛出其他未预料到的异常
            raise e

    return page


@beartype
def parse_playwright_code(code: str) -> list[ParsedPlaywrightCode]:
    # extract function calls
    if not code.startswith("page."):
        raise ValueError(
            f'Playwright action must start with "page.", but got {code}'
        )

    regex = r"\.(?![^\(\)]*\))"
    chain = re.split(regex, code)[1:]

    parsed_chain = []

    for item in chain:
        tree = ast.parse(item)
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function_name = node.func.id  # type: ignore[attr-defined]
                arguments = [
                    ast.literal_eval(arg) if isinstance(arg, ast.Str) else arg
                    for arg in node.args
                ]
                keywords = {
                    str(kw.arg): ast.literal_eval(kw.value)
                    for kw in node.keywords
                }
                funcs.append(
                    ParsedPlaywrightCode(
                        {
                            "function_name": function_name,
                            "arguments": arguments,
                            "keywords": keywords,
                        }
                    )
                )

        if len(funcs) != 1:
            raise ValueError(f"Fail to parse {item} in {code}")

        if (
            funcs[0]["function_name"]
            not in PLAYWRIGHT_LOCATORS + PLAYWRIGHT_ACTIONS
        ):
            raise ValueError(
                f"Invalid playwright code {item}, ",
                f"the function needs to be one of {PLAYWRIGHT_LOCATORS + PLAYWRIGHT_ACTIONS}",
            )

        parsed_chain.append(funcs[0])

    last_action = parsed_chain[-1]
    if last_action["function_name"] not in PLAYWRIGHT_ACTIONS:
        raise ValueError(
            f"Invalid playwright action {last_action},",
            f"the action needs to be one of {PLAYWRIGHT_ACTIONS}",
        )

    return parsed_chain


@beartype
class ActionParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


@beartype
def create_playwright_action(playwright_code: str) -> Action:
    """Main function to return individual playwright action"""
    # get the last action
    regex = r"\.(?![^\(\)]*\))"
    action = re.split(regex, playwright_code)[-1].split("(")[0]
    match action:
        case "press":
            p = r'press\((?:"|\')(.+?)(?:"|\')\)'
            match = re.search(p, playwright_code)
            if not match:
                raise ActionParsingError(
                    f"Invalid press action, required to be page.press(KEY_COMB_STR)"
                )
            key_comb = match.group(1)
            return create_key_press_action(key_comb=key_comb)
        case "scroll":
            direction = "up" if "up" in playwright_code else "down"
            return create_scroll_action(direction=direction)
        case "click":
            return create_click_action(pw_code=playwright_code)
        case "clear":
            return create_clear_action(pw_code=playwright_code)
        case "upload":
            return create_upload_action(pw_code=playwright_code)
        case "hover":
            return create_hover_action(pw_code=playwright_code)
        case "type" | "fill":
            p = r'type|fill\((?:"|\')(.+?)(?:"|\')\)'
            match = re.search(p, playwright_code)
            if not match:
                raise ActionParsingError(
                    f"Invalid type/fill action, required to be page.type(TEXT)"
                )
            text = match.group(1)
            return create_type_action(text=text, pw_code=playwright_code)
        case "select_option":
            return create_select_option_action(pw_code=playwright_code)
        case "check":
            return create_check_action(pw_code=playwright_code)
        case "goto":
            p = r'goto\((?:"|\')(.+?)(?:"|\')\)'
            match = re.search(p, playwright_code)
            if not match:
                raise ActionParsingError(
                    f"Invalid goto action, required to be page.goto(URL_STR)"
                )
            url = match.group(1)
            return create_goto_url_action(url)
        case "page_focus":
            # get the page number
            p = r"page_focus\((\d+)\)"
            match = re.search(p, playwright_code)
            if not match:
                raise ActionParsingError("page focus requires a page number")
            page_num = int(match.group(1))
            return create_page_focus_action(page_num)
        case "new_tab":
            return create_new_tab_action()
        case "go_back":
            return create_go_back_action()
        case "go_forward":
            return create_go_forward_action()
        case "page_close":
            return create_page_close_action()
        case "stop":  # page.stop(answer)
            p = r'stop\(?"(.+)?"\)'
            match = re.search(p, playwright_code)
            if not match:
                answer = ""
            else:
                answer = match.group(1)
            return create_stop_action(answer)

    raise ActionParsingError(f"Unknown playwright action {action}")


@beartype
def create_id_based_action(action_str: str) -> Optional[Action]:
    """Main function to return individual id based action"""
    action_str = action_str.strip()
    if "[" in action_str:
        action = action_str.split("[")[0].strip()
    else:
        actions = action_str.split()
        if actions:
            action = actions[0].strip()
        else:
            #raise ActionParsingError(f"No action specified: {action_str}")
            logger.warning(f"[ActionParsingError] No action specified: {action_str}")
            return None
    match action:
        case "click":
            match = re.search(r"click ?\[(\d+)\]", action_str)
            if not match:
                #raise ActionParsingError(f"Invalid click action {action_str}")
                logger.warning(f"[ActionParsingError] Invalid click action {action_str}")
                return None
            element_id = match.group(1)
            return create_click_action(element_id=element_id)
        case "clear":
            match = re.search(r"clear ?\[(\d+)\]", action_str)
            if not match:
                #raise ActionParsingError(f"Invalid clear action {action_str}")
                logger.warning(f"[ActionParsingError] Invalid clear action {action_str}")
                return None
            element_id = match.group(1)
            return create_clear_action(element_id=element_id)
        case "upload":
            # add default enter flag
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"

            match = re.search(
                r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
            )
            if not match:
                #raise ActionParsingError(f"Invalid type action {action_str}")
                logger.warning(f"[ActionParsingError] Invalid type action {action_str}")
                return None
            element_id, text, enter_flag = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if enter_flag == "1":
                text += "\n"
            return create_upload_action(text=text, element_id=element_id)
        case "hover":
            match = re.search(r"hover ?\[(\d+)\]", action_str)
            if not match:
                #raise ActionParsingError(f"Invalid hover action {action_str}")
                logger.warning(f"[ActionParsingError] Invalid hover action {action_str}")
                return None
            element_id = match.group(1)
            return create_hover_action(element_id=element_id)
        case "type":
            # add default enter flag
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"

            match = re.search(
                r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
            )
            if not match:
                #raise ActionParsingError(f"Invalid type action {action_str}")
                logger.warning(f"[ActionParsingError] Invalid type action {action_str}")
                return None
            element_id, text, enter_flag = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            # if enter_flag == "1":
            #     text += "\n"
            return create_type_action(text=text, element_id=element_id)
        case "press":
            match = re.search(r"press ?\[(.+)\]", action_str)
            if not match:
                #raise ActionParsingError(f"Invalid press action {action_str}")
                logger.warning(f"[ActionParsingError] Invalid press action {action_str}")
                return None
            key_comb = match.group(1)
            return create_key_press_action(key_comb=key_comb)
        case "scroll":
            # up or down
            match = re.search(r"scroll ?\[?(up|down)\]?", action_str)
            if not match:
                #raise ActionParsingError(f"Invalid scroll action {action_str}")
                logger.warning(f"[ActionParsingError] Invalid scroll action {action_str}")
                return None
            direction = match.group(1)
            return create_scroll_action(direction=direction)
        case "goto":
            match = re.search(r"goto ?\[(.+)\]", action_str)
            if not match:
                #raise ActionParsingError(f"Invalid goto action {action_str}")
                logger.warning(f"[ActionParsingError] Invalid goto action {action_str}")
                return None
            url = match.group(1)
            return create_goto_url_action(url=url)
        case "new_tab":
            return create_new_tab_action()
        case "go_back":
            return create_go_back_action()
        case "go_forward":
            return create_go_forward_action()
        case "tab_focus":
            match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
            if not match:
                # raise ActionParsingError(
                #     f"Invalid tab_focus action {action_str}"
                # )
                logger.warning(f"[ActionParsingError] Invalid tab_focus action {action_str}")
                return None
            page_number = int(match.group(1))
            return create_page_focus_action(page_number)
        case "close_tab":
            return create_page_close_action()
        case "stop":  # stop answer
            match = re.search(r"stop ?\[(.+)\]", action_str)
            if not match:  # some tasks don't require an answer
                answer = ""
            else:
                answer = match.group(1)
            return create_stop_action(answer)

    #raise ActionParsingError(f"Invalid action {action_str}")
    logger.warning(f"[ActionParsingError] Invalid action {action_str}")
    return None


@beartype
def create_coords_based_action(action_str: str) -> Optional[Action]:
    """
    Parses a string representing a coordinate-based action and returns a valid Action object.
    The coordinate-based actions are derived from the JSON schema provided.
    """
    action_str = action_str.strip()
    action_name = action_str.split('[')[0].strip()

    # Start with a default 'none' action
    action = create_none_action()

    try:
        match action_name:
            case "click":
                match = re.search(r"click\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", action_str)
                if not match: return None
                coords = np.array([float(match.group(1)), float(match.group(2))], dtype=np.float32)
                action.update({"action_type": ActionTypes.MOUSE_CLICK, "coords": coords})
                return action

            case "double_click":
                match = re.search(r"double_click\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", action_str)
                if not match: return None
                coords = np.array([float(match.group(1)), float(match.group(2))], dtype=np.float32)
                action.update({"action_type": ActionTypes.DOUBLE_CLICK, "coords": coords})
                return action

            case "type":
                match = re.search(r"type\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*\[(.*?)\](?: ?\[(0|1)\])?", action_str)
                if not match: return None
                coords = np.array([float(match.group(1)), float(match.group(2))], dtype=np.float32)
                content = match.group(3)
                press_enter = match.group(4) if match.group(4) else 0
                # if press_enter is None or press_enter == '1':  # Default to pressing enter
                #     content += "\n"
                action.update({"action_type": ActionTypes.TYPE, "coords": coords, "text": _keys2ids(content), "press_enter_after": press_enter})
                return action

            case "hover" | "move_to":
                match = re.search(r"(?:hover|move_to)\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", action_str)
                if not match: return None
                coords = np.array([float(match.group(1)), float(match.group(2))], dtype=np.float32)
                action.update({"action_type": ActionTypes.MOUSE_HOVER, "coords": coords})
                return action

            case "press":
                match = re.search(r"press\s*\[(.*?)\]", action_str)
                if not match: return None
                return create_key_press_action(key_comb=match.group(1))

            case "scroll":
                # Support all four directions (up/down/left/right) with optional coordinates
                # Format: scroll[direction][distance] or scroll[direction][distance][x,y]
                match = re.search(r"scroll\s*\[(up|down|left|right)\]\s*\[(\d+)\](?:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\])?", action_str)
                if not match: return None
                direction, distance = match.group(1), match.group(2)
                
                # Check if coordinates are provided
                coords = None
                if match.group(3) and match.group(4):
                    coords = np.array([float(match.group(3)), float(match.group(4))], dtype=np.float32)
                
                # All directions use SCROLL action type
                action.update({"action_type": ActionTypes.SCROLL, "direction": direction, "nth": int(distance), "coords": coords})
                return action

            case "browser_select_option":
                match = re.search(r"browser_select_option\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*\[(.*?)\]", action_str)
                if not match: return None
                coords = np.array([float(match.group(1)), float(match.group(2))], dtype=np.float32)
                option = match.group(3)
                action.update({"action_type": ActionTypes.COORDS_SELECT_OPTION, "coords": coords, "element_name": option})
                return action
            
            case "wait":
                match = re.search(r"wait\s*\[(\d+)\]", action_str)
                if not match: return None
                seconds = int(match.group(1))
                action.update({"action_type": ActionTypes.WAIT, "nth": seconds})
                return action
            
            case "find":
                # Format: find[search_term]
                match = re.search(r"find\s*\[(.*?)\]", action_str)
                if not match: return None
                search_term = match.group(1)
                action.update({"action_type": ActionTypes.FIND, "search_term": search_term})
                return action
            
            case "get_file":
                # Format: get_file[x,y]
                match = re.search(r"get_file\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", action_str)
                if not match: return None
                coords = np.array([float(match.group(1)), float(match.group(2))], dtype=np.float32)
                action.update({"action_type": ActionTypes.GET_FILE, "coords": coords})
                return action

            case "goto":
                match = re.search(r"goto\s*\[(.*?)\]", action_str)
                if not match: return None
                return create_goto_url_action(url=match.group(1))

            case "new_tab":
                return create_new_tab_action()

            case "tab_focus":
                match = re.search(r"tab_focus\s*\[(\d+)\]", action_str)
                if not match: return None
                return create_page_focus_action(int(match.group(1)))

            case "close_tab":
                return create_page_close_action()

            case "go_back":
                return create_go_back_action()

            case "go_forward":
                return create_go_forward_action()

            case "stop":
                match = re.search(r"stop\s*\[(.*?)\]", action_str)
                answer = match.group(1) if match else ""
                return create_stop_action(answer)

            case _:
                logger.warning(f"[ActionParsingError] Unknown or malformed coordinate-based action: {action_str}")
                return None
    except Exception as e:
        logger.error(f"[ActionParsingError] Failed to parse action '{action_str}': {e}")
        return None

@beartype
@with_timeout_legacy(2 * 60)
async def aexecute_action_coords(
    action: Action,
    page: APage,
    browser_ctx: ABrowserContext,
    obseration_processor: AsyncImageObservationProcessor  # Added processor argument
) -> APage:
    """
    Executes an Action object, assuming it was created from a coordinate-based string.
    This function uses direct coordinate manipulation and now validates coordinates against
    interactable elements via the observation processor.
    """
    num_tabs_before = len(browser_ctx.pages)
    action_type = action["action_type"]

    match action_type:
        case ActionTypes.MOUSE_CLICK:
            # Get the coordinates
            x, y = action["coords"]
            # Convert to standard Python float before passing to the function
            await page.mouse.click(float(x), float(y))
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.DOUBLE_CLICK:
            x, y = action["coords"]
            await page.mouse.dblclick(float(x), float(y))
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.MOUSE_HOVER | ActionTypes.MOVE_TO:
            x, y = action["coords"]
            await page.mouse.move(float(x), float(y))
            await page.wait_for_timeout(500)
        case ActionTypes.TYPE:
            if np.any(action["coords"]):
                x, y = action["coords"]
                # Also convert here
                #连点三次模拟全选
                await page.mouse.click(float(x), float(y), click_count=3)
            await page.wait_for_timeout(500)
            text = "".join([_id2key[key] for key in action["text"]])
            await aexecute_key_press("Backspace", page)
            await page.wait_for_timeout(500)
            await aexecute_keyboard_type(text, page)
            await page.wait_for_timeout(500)
            if str(action["press_enter_after"]) == "1":
                await aexecute_key_press("Enter", page)
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.SCROLL:
            distance = int(action["nth"])
            direction = action["direction"]
            coords = action.get("coords", None)
            
            if coords is not None and np.any(coords):
                # Scroll at specific coordinates using locate_scroll.js
                x, y = float(coords[0]), float(coords[1])
                scroll_js_path = Path(__file__).parent / "javascript" / "action_js" / "locate_scroll.js"
                with open(scroll_js_path, "r") as f:
                    scroll_js = f.read()
                await page.evaluate(scroll_js, {"x": x, "y": y, "direction": direction, "distance": distance})
            else:
                # Scroll the entire page (vertical or horizontal based on direction)
                if direction in ["up", "down"]:
                    scroll_y = -distance if direction == "up" else distance
                    await page.evaluate(f"window.scrollBy(0, {scroll_y})")
                else:  # left or right
                    scroll_x = -distance if direction == "left" else distance
                    await page.evaluate(f"window.scrollBy({scroll_x}, 0)")
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.KEY_PRESS:
            await aexecute_key_press(action["key_comb"], page)
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.GOTO_URL:
            await page.goto(action["url"], wait_until="networkidle")
            
        case ActionTypes.GO_BACK:
            if len(browser_ctx.pages) > 1:
                await page.go_back()
        
        case ActionTypes.GO_FORWARD:
            await page.go_forward()

        case ActionTypes.NEW_TAB:
            page = await browser_ctx.new_page()
            await page.goto("https://www.baidu.com", wait_until="networkidle")


        case ActionTypes.PAGE_CLOSE:
            if len(browser_ctx.pages) > 1:
                await page.close()
                if len(browser_ctx.pages) > 0:
                    page = browser_ctx.pages[-1]
                else:
                    page = await browser_ctx.new_page()
                    await page.goto("https://www.baidu.com", wait_until="networkidle")

            

        case ActionTypes.PAGE_FOCUS:
            page = browser_ctx.pages[action["page_number"]]
            await page.bring_to_front()
        
        case ActionTypes.COORDS_SELECT_OPTION:
            x, y = float(action["coords"][0]), float(action["coords"][1])
            option_text = action["element_name"]

            if x is not None and y is not None and option_text:
                try:
                    # 步骤 1: 点击坐标，强制打开下拉菜单
                    # 这确保了即使坐标指向的是一个包裹 <select> 的 <div>，也能触发下拉。
                    await page.mouse.click(x, y)
                    # 给予一点时间让系统UI渲染出来
                    await page.wait_for_timeout(300)

                    # 步骤 2: 使用 page.select_option 选择
                    # 现在下拉菜单已经打开，Playwright可以更好地识别选项。
                    # 我们需要找到被点击的 <select> 元素。
                    # Playwright 没有直接从坐标获取元素的方法，但我们可以用JS找到它，然后传递给 locator。

                    # 用JS找到精确的元素
                    target_selector = await page.evaluate(f'''() => {{
                        const el = document.elementFromPoint({x}, {y});
                        if (!el) return null;

                        // 向上查找，直到找到 <select> 或 body
                        let target = el;
                        for (let i=0; i<5; i++) {{ // 最多向上找5层
                            if (target.tagName === 'SELECT') break;
                            if (!target.parentElement || target.parentElement.tagName === 'BODY') break;
                            target = target.parentElement;
                        }}

                        // 如果没找到 <select>，就用原始点击的元素
                        if (target.tagName !== 'SELECT') {{
                            target = el;
                        }}

                        // 生成一个唯一的CSS选择器
                        if (target.id) return `#${{target.id}}`;
                        if (target.name) return `[name="${{target.name}}"]`;
                        // 备用选择器，可能不稳定但聊胜于无
                        return `*:nth-of-type(1)`;
                    }}''')

                    if target_selector:
                        # 使用生成的选择器和标签文本进行选择
                        await page.select_option(target_selector, label=option_text, timeout=3000)
                    else:
                        raise Exception("Could not generate a selector for the element at the coordinates.")

                except Exception as e:
                    # 如果上述方法失败，回到最原始但可能有效的JS方法
                    logger.warning(f"Playwright's select_option failed: {e}. Falling back to JS execution.")
                    try:
                        js_code = f'''
                        async (optionText) => {{
                            let el = document.elementFromPoint({x}, {y});
                            if (!el) return false;

                            // 如果点击的不是 select，向上查找
                            if (el.tagName !== 'SELECT') {{
                                const parentSelect = el.closest('select');
                                if (parentSelect) el = parentSelect;
                            }}

                            if (el && el.tagName === 'SELECT') {{
                                for (const option of el.options) {{
                                    // 1. 优先尝试完全匹配文本
                                    if (option.text.trim() === optionText.trim()) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                    // 2. 尝试包含匹配文本
                                    if (option.text.trim().includes(optionText.trim())) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                    // 3. 尝试匹配 value (如果 optionText 恰好是 value)
                                    if (option.value.trim() === optionText.trim()) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                }}
                            }}
                            return false;
                        }}
                        '''
                        success = await page.evaluate(js_code, option_text)
                        if not success:
                            error_msg = f"JS fallback for select_option also failed for option: {option_text}"
                            logger.error(error_msg)

                    except Exception as e2:
                        error_msg = f"JS fallback for select_option threw an error: {e2}"
                        logger.error(error_msg)
            else:
                error_msg = f"select_option action missing coords or option text. Action: {action}"
                logger.warning(error_msg)
            await page.wait_for_timeout(2 * 1000)
        case ActionTypes.WAIT:
            await page.wait_for_timeout(action["nth"] * 1000)
        
        case ActionTypes.FIND:
            search_term = action["search_term"]
            occurrence = action.get("occurrence", 0)  # 默认第一个匹配项
            highlight = action.get("highlight", True)  # 默认高亮
            
            logger.info(f"正在搜索: '{search_term}', 目标是第 {occurrence + 1} 个匹配项...")
            
            try:
                # 使用 get_by_text 定位所有匹配的元素
                locator = page.get_by_text(search_term, exact=False)
                
                # 检查找到了多少个匹配项
                count = await locator.count()
                if count == 0:
                    logger.warning(f"Text '{search_term}' not found on the page")
                else:
                    logger.info(f"找到了 {count} 个匹配项。")
                    
                    # 检查指定的索引是否有效
                    if occurrence >= count:
                        logger.warning(f"Error: Trying to focus on match #{occurrence + 1} but only found {count} matches.")
                    else:
                        # 获取指定索引的那个元素
                        target_element = locator.nth(occurrence)
                        
                        # 滚动页面，使其进入视图
                        logger.info("滚动页面至目标元素...")
                        await target_element.scroll_into_view_if_needed()
                        
                        # 可选：高亮元素
                        if highlight:
                            logger.info("高亮显示目标元素...")
                            try:
                                await target_element.evaluate("""
                                    (element) => {
                                        const originalStyle = element.style.cssText;
                                        element.style.cssText = originalStyle + '; background-color: yellow !important; transition: background-color 0.3s;';
                                        setTimeout(() => {
                                            element.style.cssText = originalStyle;
                                        }, 2000);
                                    }
                                """)
                            except Exception as e:
                                logger.warning(f"高亮元素失败: {e}")
                        
                        logger.info(f"🔍 Successfully found and focused on text '{search_term}' (match #{occurrence + 1})")
                        await page.wait_for_timeout(500)
            except Exception as e:
                logger.error(f"Error occurred while finding text: {e}", exc_info=True)
            
            await page.wait_for_timeout(1000)
        
        case ActionTypes.GET_FILE:
            x, y = float(action["coords"][0]), float(action["coords"][1])
            logger.info(f"正在尝试从坐标 ({x}, {y}) 获取文件...")
            
            try:
                # 读取 get_file.js
                get_file_js_path = Path(__file__).parent / "javascript" / "action_js" / "get_file.js"
                with open(get_file_js_path, "r") as f:
                    get_file_js = f.read()
                
                # 执行 JavaScript 获取文件信息
                result = await page.evaluate(get_file_js, [x, y])
                
                if result is None:
                    logger.warning(f"未在坐标 ({x}, {y}) 找到可下载的文件或图片")
                else:
                    file_type = result.get("type")
                    file_url = result.get("url")
                    logger.info(f"找到 {file_type} 文件: {file_url}")
                    
                    # 如果是文件链接，触发下载
                    if file_type not in ["image"]:
                        # 点击链接以触发下载
                        await page.mouse.click(x, y)
                        logger.info(f"已点击链接触发下载")
                    else:
                        logger.info(f"找到图片: {file_url}")
                        # 对于图片，可以选择性地记录 URL 或进行其他操作
                    
                    await page.wait_for_timeout(2000)
            except Exception as e:
                logger.error(f"获取文件时发生错误: {e}", exc_info=True)

        case ActionTypes.NONE | ActionTypes.STOP:
            pass
        
        case _:
            raise ValueError(f"aexecute_action_coords received unhandled action type: {action_type}")
    # --- 原始的 match 逻辑结束 ---

    # 检查是否有新标签页被打开（例如，点击链接后在新窗口打开）
    # 1. 处理可能新打开的标签页
    if len(browser_ctx.pages) > num_tabs_before:
        page = browser_ctx.pages[-1]
        await page.bring_to_front()
    # 2. 在动作执行后等待页面稳定
    try:
        await page.wait_for_load_state(
            "networkidle", timeout=ACTION_POST_WAIT_TIMEOUT
        )
    except PlaywrightTimeoutError:
        # 如果没有发生导航，这是预期的行为。
        pass
    except Exception as e:
        if page.is_closed():
            logger.warning(f"页面在执行坐标动作 '{action_type}' 后被关闭。重新分配到一个有效的页面。")
            if len(browser_ctx.pages) > 0:
                page = browser_ctx.pages[-1]
            else:
                page = await browser_ctx.new_page()
                await page.goto("https://www.baidu.com", wait_until="networkidle")

        else:
            raise e

    return page

@beartype
def execute_action_coords(
    action: Action,
    page: Page,
    browser_ctx: BrowserContext,
    obseration_processor: ImageObservationProcessor,  # Changed to the sync processor
) -> Page:
    """
    Executes a synchronous Action object, assuming it was created from a coordinate-based string.
    This function uses direct coordinate manipulation and validates coordinates against
    interactable elements via the observation processor.
    """
    num_tabs_before = len(browser_ctx.pages)
    action_type = action["action_type"]

    # --- 原始的验证和 match action_type 逻辑保持完全不变 ---
    # ... (从 coord_actions_to_validate = { ... 到 raise ValueError(...) 的所有代码都保持原样)
    coord_actions_to_validate = {
        ActionTypes.MOUSE_CLICK,
        ActionTypes.DOUBLE_CLICK,
        ActionTypes.TYPE,  # Because it clicks first to focus
        ActionTypes.COORDS_SELECT_OPTION,
        ActionTypes.GET_FILE  # Requires coords to locate file/image
    }

    # if action_type in coord_actions_to_validate:
    #     if np.any(action["coords"]):
    #         x, y = action["coords"]
    #         # Validate that the coordinates point to an interactable element
    #         # Assumes the sync processor has this method
    #         if not obseration_processor.is_coords_on_interactable_element(float(x), float(y)):
    #             raise ValueError(
    #                 f"Action '{action_type.name}' at ({x:.0f}, {y:.0f}) is invalid: "
    #                 "The coordinates provided in the action do not correspond to any known interactable element in the last screenshot."
    #             )
    #     # If type action has no coords, it types in the currently focused element, which is valid.
    #     elif action_type != ActionTypes.TYPE:
    #          raise ValueError(f"Action '{action_type.name}' requires coordinates, but none were provided.")

    match action_type:
        case ActionTypes.MOUSE_CLICK:
            x, y = action["coords"]
            page.mouse.click(float(x), float(y))
        
        case ActionTypes.DOUBLE_CLICK:
            x, y = action["coords"]
            page.mouse.dblclick(float(x), float(y))

        case ActionTypes.MOUSE_HOVER | ActionTypes.MOVE_TO:
            x, y = action["coords"]
            page.mouse.move(float(x), float(y))

        case ActionTypes.TYPE:
            if np.any(action["coords"]):
                x, y = action["coords"]
                page.mouse.click(float(x), float(y))
            text = "".join([_id2key[key] for key in action["text"]])
            execute_keyboard_type(text, page)

        case ActionTypes.SCROLL:
            distance = int(action["nth"])
            direction = action["direction"]
            coords = action.get("coords", None)
            
            if coords is not None and np.any(coords):
                # Scroll at specific coordinates using locate_scroll.js
                x, y = float(coords[0]), float(coords[1])
                scroll_js_path = Path(__file__).parent / "javascript" / "action_js" / "locate_scroll.js"
                with open(scroll_js_path, "r") as f:
                    scroll_js = f.read()
                page.evaluate(scroll_js, {"x": x, "y": y, "direction": direction, "distance": distance})
            else:
                # Scroll the entire page (vertical or horizontal based on direction)
                if direction in ["up", "down"]:
                    scroll_y = -distance if direction == "up" else distance
                    page.evaluate(f"window.scrollBy(0, {scroll_y})")
                else:  # left or right
                    scroll_x = -distance if direction == "left" else distance
                    page.evaluate(f"window.scrollBy({scroll_x}, 0)")

        case ActionTypes.KEY_PRESS:
            execute_key_press(action["key_comb"], page)
        
        case ActionTypes.GOTO_URL:
            page.goto(action["url"])
            
        case ActionTypes.GO_BACK:
            if len(browser_ctx.pages) > 1:
                page.go_back()
        
        case ActionTypes.GO_FORWARD:
            page.go_forward()

        case ActionTypes.NEW_TAB:
            page = browser_ctx.new_page()
            page.goto("https://www.baidu.com", wait_until="networkidle")


        case ActionTypes.PAGE_CLOSE:
            if len(browser_ctx.pages) > 1:
                page.close()
                if len(browser_ctx.pages) > 0:
                    page = browser_ctx.pages[-1]
                else:
                    page = browser_ctx.new_page()
                    page.goto("https://www.baidu.com", wait_until="networkidle")
            

        case ActionTypes.PAGE_FOCUS:
            page = browser_ctx.pages[action["page_number"]]
            page.bring_to_front()
        
        case ActionTypes.COORDS_SELECT_OPTION:
            x, y = float(action["coords"][0]), float(action["coords"][1])
            option_text = action["element_name"]

            if x is not None and y is not None and option_text:
                try:
                    # 步骤 1: 点击坐标，强制打开下拉菜单
                    # 这确保了即使坐标指向的是一个包裹 <select> 的 <div>，也能触发下拉。
                    # page.mouse.click(x, y)
                    # # 给予一点时间让系统UI渲染出来
                    # page.wait_for_timeout(300)

                    # 步骤 2: 使用 page.select_option 选择
                    # 现在下拉菜单已经打开，Playwright可以更好地识别选项。
                    # 我们需要找到被点击的 <select> 元素。
                    # Playwright 没有直接从坐标获取元素的方法，但我们可以用JS找到它，然后传递给 locator。

                    # 用JS找到精确的元素
                    target_selector = page.evaluate(f'''() => {{
                        const el = document.elementFromPoint({x}, {y});
                        if (!el) return null;

                        // 向上查找，直到找到 <select> 或 body
                        let target = el;
                        for (let i=0; i<5; i++) {{ // 最多向上找5层
                            if (target.tagName === 'SELECT') break;
                            if (!target.parentElement || target.parentElement.tagName === 'BODY') break;
                            target = target.parentElement;
                        }}

                        // 如果没找到 <select>，就用原始点击的元素
                        if (target.tagName !== 'SELECT') {{
                            target = el;
                        }}

                        // 生成一个唯一的CSS选择器
                        if (target.id) return `#${{target.id}}`;
                        if (target.name) return `[name="${{target.name}}"]`;
                        // 备用选择器，可能不稳定但聊胜于无
                        return `*:nth-of-type(1)`;
                    }}''')

                    if target_selector:
                        # 使用生成的选择器和标签文本进行选择
                        page.select_option(target_selector, label=option_text, timeout=3000)
                    else:
                        raise Exception("Could not generate a selector for the element at the coordinates.")

                except Exception as e:
                    # 如果上述方法失败，回到最原始但可能有效的JS方法
                    logger.warning(f"Playwright's select_option failed: {e}. Falling back to JS execution.")
                    try:
                        js_code = f'''
                        (optionText) => {{
                            let el = document.elementFromPoint({x}, {y});
                            if (!el) return false;

                            // 如果点击的不是 select，向上查找
                            if (el.tagName !== 'SELECT') {{
                                const parentSelect = el.closest('select');
                                if (parentSelect) el = parentSelect;
                            }}

                            if (el && el.tagName === 'SELECT') {{
                                for (const option of el.options) {{
                                    // 1. 优先尝试完全匹配文本
                                    if (option.text.trim() === optionText.trim()) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                    // 2. 尝试包含匹配文本
                                    if (option.text.trim().includes(optionText.trim())) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                    // 3. 尝试匹配 value (如果 optionText 恰好是 value)
                                    if (option.value.trim() === optionText.trim()) {{
                                        el.value = option.value;
                                        el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        return true;
                                    }}
                                }}
                            }}
                            return false;
                        }}
                        '''
                        success = page.evaluate(js_code, option_text)
                        if not success:
                            error_msg = f"JS fallback for select_option also failed for option: {option_text}"
                            logger.error(error_msg)

                    except Exception as e2:
                        error_msg = f"JS fallback for select_option threw an error: {e2}"
                        logger.error(error_msg)
            else:
                error_msg = f"select_option action missing coords or option text. Action: {action}"
                logger.warning(error_msg)
            page.wait_for_timeout(2 * 1000)

        case ActionTypes.WAIT:
            page.wait_for_timeout(action["nth"] * 1000)
        
        case ActionTypes.FIND:
            search_term = action["search_term"]
            occurrence = action.get("occurrence", 0)  # 默认第一个匹配项
            highlight = action.get("highlight", True)  # 默认高亮
            
            logger.info(f"正在搜索: '{search_term}', 目标是第 {occurrence + 1} 个匹配项...")
            
            try:
                # 使用 get_by_text 定位所有匹配的元素
                locator = page.get_by_text(search_term, exact=False)
                
                # 检查找到了多少个匹配项
                count = locator.count()
                if count == 0:
                    logger.warning(f"Text '{search_term}' not found on the page")
                else:
                    logger.info(f"找到了 {count} 个匹配项。")
                    
                    # 检查指定的索引是否有效
                    if occurrence >= count:
                        logger.warning(f"Error: Trying to focus on match #{occurrence + 1} but only found {count} matches.")
                    else:
                        # 获取指定索引的那个元素
                        target_element = locator.nth(occurrence)
                        
                        # 滚动页面，使其进入视图
                        logger.info("滚动页面至目标元素...")
                        target_element.scroll_into_view_if_needed()
                        
                        # 可选：高亮元素
                        if highlight:
                            logger.info("高亮显示目标元素...")
                            try:
                                target_element.evaluate("""
                                    (element) => {
                                        const originalStyle = element.style.cssText;
                                        element.style.cssText = originalStyle + '; background-color: yellow !important; transition: background-color 0.3s;';
                                        setTimeout(() => {
                                            element.style.cssText = originalStyle;
                                        }, 2000);
                                    }
                                """)
                            except Exception as e:
                                logger.warning(f"高亮元素失败: {e}")
                        
                        logger.info(f"🔍 Successfully found and focused on text '{search_term}' (match #{occurrence + 1})")
                        page.wait_for_timeout(500)
            except Exception as e:
                logger.error(f"Error occurred while finding text: {e}", exc_info=True)
            
            page.wait_for_timeout(1000)
        
        case ActionTypes.GET_FILE:
            x, y = float(action["coords"][0]), float(action["coords"][1])
            logger.info(f"正在尝试从坐标 ({x}, {y}) 获取文件...")
            
            try:
                # 读取 get_file.js
                get_file_js_path = Path(__file__).parent / "javascript" / "action_js" / "get_file.js"
                with open(get_file_js_path, "r") as f:
                    get_file_js = f.read()
                
                # 执行 JavaScript 获取文件信息
                result = page.evaluate(get_file_js, [x, y])
                
                if result is None:
                    logger.warning(f"未在坐标 ({x}, {y}) 找到可下载的文件或图片")
                else:
                    file_type = result.get("type")
                    file_url = result.get("url")
                    logger.info(f"找到 {file_type} 文件: {file_url}")
                    
                    # 如果是文件链接，触发下载
                    if file_type not in ["image"]:
                        # 点击链接以触发下载
                        page.mouse.click(x, y)
                        logger.info(f"已点击链接触发下载")
                    else:
                        logger.info(f"找到图片: {file_url}")
                        # 对于图片，可以选择性地记录 URL 或进行其他操作
                    
                    page.wait_for_timeout(2000)
            except Exception as e:
                logger.error(f"获取文件时发生错误: {e}", exc_info=True)

        case ActionTypes.NONE | ActionTypes.STOP:
            pass
        
        case _:
            raise ValueError(f"execute_action_coords received unhandled action type: {action_type}")
    # --- 原始的 match 逻辑结束 ---


    # --- 新增的逻辑 (与 execute_action 中的修改完全相同) ---

    # 检查是否有新标签页被打开
    # 检查是否有新标签页被打开（例如，点击链接后在新窗口打开）
    # 1. 处理可能新打开的标签页
    if len(browser_ctx.pages) > num_tabs_before:
        current_page_num = 0
        for page_ind, page_i in enumerate(browser_ctx.pages):
            if page_i == page:
                current_page_num = page_ind
                break
        page = browser_ctx.pages[current_page_num + 1]
        #logger.info(f"aexecute_action_coords: 新打开的标签页: {page.url}")
        page.bring_to_front()

    # 关键修改：在操作后等待页面稳定
    try:
        page.wait_for_load_state("networkidle", timeout=domcontentloaded_timeout)
    except Exception:
        # 如果没有发生导航，则忽略超时
        pass

    return page