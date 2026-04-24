# 浏览器动作测试用例

## 概述

本文档说明如何运行新增和修改的浏览器动作测试用例。

## 新增的动作

### 1. `find` - 搜索文本
在页面上全局搜索文本并滚动到匹配项。

**格式**: `find [search_term]`

**示例**: `find [登录]`

**参数**:
- `search_term`: 要搜索的文本

### 2. `get_file` - 获取文件/图片
从指定坐标获取文件或图片。

**格式**: `get_file [x, y]`

**示例**: `get_file [500, 300]`

**参数**:
- `coords`: 点击坐标 [x, y]

## 修改的动作

### `scroll` - 滚动页面（支持四个方向 + 可选坐标）

**格式**: 
- 基本滚动: `scroll [direction] [distance]`
- 在特定位置滚动: `scroll [direction] [distance] [x, y]`

**方向支持**:
- `up`: 向上滚动
- `down`: 向下滚动
- `left`: 向左滚动
- `right`: 向右滚动

**示例**:
```python
# 基本滚动
scroll [down] [500]      # 向下滚动 500 像素
scroll [up] [300]        # 向上滚动 300 像素
scroll [left] [200]      # 向左滚动 200 像素
scroll [right] [400]     # 向右滚动 400 像素

# 在特定位置滚动
scroll [down] [500] [600, 400]    # 在坐标 (600, 400) 处向下滚动 500 像素
scroll [right] [300] [800, 500]   # 在坐标 (800, 500) 处向右滚动 300 像素
```

## 运行测试

### 方法 1: 直接运行测试脚本

```bash
cd Agent-R1/verl/recipe/webagent_fully_async_policy/browser_env

python tool_env.py test
```

### 方法 2: 在 Python 中导入测试函数

```python
from recipe.webagent_fully_async_policy.browser_env.tool_env import test_new_actions, test_web_browser_tool_conversion

# 测试动作解析
test_new_actions()

# 测试 WebBrowserTool 转换
test_web_browser_tool_conversion()
```

## 测试内容

### `test_new_actions()` - 测试动作解析

测试 `create_coords_based_action` 函数是否能正确解析新增和修改的动作：

1. ✅ `find` 动作解析
2. ✅ `get_file` 动作解析
3. ✅ `scroll [up]` 垂直向上滚动
4. ✅ `scroll [down]` 垂直向下滚动
5. ✅ `scroll [left]` 水平向左滚动
6. ✅ `scroll [right]` 水平向右滚动
7. ✅ `scroll [down]` 带坐标的垂直滚动
8. ✅ `scroll [right]` 带坐标的水平滚动

### `test_web_browser_tool_conversion()` - 测试工具转换

测试 `WebBrowserTool.convert_action_to_string_coords` 函数是否能正确转换动作参数：

1. ✅ `find` 动作转换
2. ✅ `get_file` 动作转换（含坐标反缩放）
3. ✅ `scroll` 四个方向的动作转换
4. ✅ `scroll` 带坐标的动作转换

## 测试输出示例

```
============================================================
测试新增和修改的动作
============================================================

测试 1: find 动作
----------------------------------------
输入: find [登录]
解析结果: {'action_type': 24, 'search_term': '登录', ...}
✓ find 动作测试通过

测试 2: get_file 动作
----------------------------------------
输入: get_file [500, 300]
解析结果: {'action_type': 25, 'coords': array([500., 300.], dtype=float32), ...}
✓ get_file 动作测试通过

...

============================================================
所有测试通过！✓
============================================================
```

## ActionTypes 枚举值

```python
SCROLL = 1           # 滚动（支持 up/down/left/right）
WAIT = 23            # 等待
FIND = 24            # 搜索文本
GET_FILE = 25        # 获取文件/图片
```

## 注意事项

1. **HSCROLL 已删除**: 水平滚动现在统一使用 `SCROLL` 类型，通过 `direction` 参数区分（left/right）
2. **坐标缩放**: 所有坐标参数都会经过 `convert_action_to_string_coords` 进行缩放转换
3. **可选坐标**: `scroll` 动作的 `coords` 参数是可选的，如果不提供则滚动整个页面

## 相关文件

- `actions.py` - 动作定义和执行逻辑
- `action_space_json.py` - 动作空间的 JSON 定义
- `web_browser_tool.py` - WebBrowserTool 工具类
- `tool_env.py` - 本测试文件

## 问题反馈

如果测试失败或发现问题，请检查：
1. ActionTypes 枚举值是否正确
2. 动作字符串格式是否符合规范
3. JavaScript 文件（locate_scroll.js, get_file.js）是否存在
