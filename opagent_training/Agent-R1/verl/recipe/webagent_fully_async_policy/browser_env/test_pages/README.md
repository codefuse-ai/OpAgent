# 浏览器动作测试页面

本目录包含用于测试新增和修改的浏览器动作的HTML测试页面。

## 📁 文件结构

```
test_pages/
├── test_scroll.html       # 滚动测试页面（四个方向 + 坐标滚动）
├── test_find.html         # 文本搜索测试页面
├── test_get_file.html     # 文件获取测试页面
├── screenshots/           # 测试截图保存目录（自动创建）
└── README.md             # 本文件
```

## 🧪 测试页面说明

### 1. test_scroll.html - 滚动测试
**测试功能**: 
- ✅ 垂直滚动 (up/down)
- ✅ 水平滚动 (left/right)
- ✅ 在特定坐标位置滚动
- ✅ 可滚动容器内的滚动

**页面特点**:
- 页面尺寸: 3000x3000 像素
- 包含多个可滚动区域
- 有标记点显示坐标位置
- 实时显示当前滚动位置

**测试动作示例**:
```python
scroll [down] [500]           # 向下滚动500像素
scroll [up] [300]             # 向上滚动300像素
scroll [left] [400]           # 向左滚动400像素
scroll [right] [600]          # 向右滚动600像素
scroll [down] [500] [300, 400]  # 在坐标(300,400)处向下滚动
```

### 2. test_find.html - 文本搜索测试
**测试功能**:
- ✅ 搜索可见文本
- ✅ 搜索隐藏文本（需滚动）
- ✅ 中文文本搜索
- ✅ 英文文本搜索
- ✅ 自动滚动到匹配位置
- ✅ 文本高亮显示

**页面特点**:
- 包含多个"目标文本"实例
- 部分内容需要滚动才能看到
- 包含各种常用中英文关键词

**测试动作示例**:
```python
find [目标文本]    # 搜索中文
find [登录]       # 搜索常用词
find [确定]       # 搜索隐藏内容
find [Login]     # 搜索英文
```

### 3. test_get_file.html - 文件获取测试
**测试功能**:
- ✅ 识别 PDF 文件
- ✅ 识别 Word 文档 (.docx)
- ✅ 识别 Excel 表格 (.xlsx)
- ✅ 识别 PowerPoint (.pptx)
- ✅ 识别压缩文件 (.zip, .rar)
- ✅ 识别图片文件
- ✅ 识别文本文件 (.txt, .csv)

**页面特点**:
- 包含多种文件类型的下载链接
- 每个文件卡片标注了建议测试坐标
- 包含图片画廊区域
- 有测试检查清单

**测试动作示例**:
```python
get_file [230, 470]   # 获取 PDF 文件
get_file [480, 400]   # 获取 Word 文档
get_file [150, 400]   # 获取图片（需先滚动）
```

## 🚀 运行测试

### 方法 1: 单元测试（不需要浏览器）
```bash
cd Agent-R1/verl/recipe/webagent_fully_async_policy/browser_env

python tool_env.py test
```

这将运行快速的单元测试，验证动作解析逻辑。

### 方法 2: 浏览器交互测试（需要浏览器）
```bash
python tool_env.py browser_test
```

这将：
1. 加载每个测试HTML页面
2. 执行一系列测试动作
3. 在每个步骤后保存截图
4. 生成测试报告

**前置条件**:
- 需要有可用的浏览器 WebSocket 端点
- 检查 `BROWSER_OUTPUT_PATH` 环境变量配置

### 测试输出

浏览器测试会生成以下截图（保存在 `screenshots/` 目录）:

**Scroll 测试**:
- `01_scroll_initial.png` - 初始状态
- `02_scroll_down.png` - 向下滚动后
- `03_scroll_right.png` - 向右滚动后
- `04_scroll_up.png` - 向上滚动后
- `05_scroll_left.png` - 向左滚动后
- `06_scroll_at_coords.png` - 在特定位置滚动后

**Find 测试**:
- `07_find_initial.png` - 初始状态
- `08_find_target.png` - 搜索"目标文本"后
- `09_find_login.png` - 搜索"登录"后
- `10_find_hidden.png` - 搜索隐藏内容后

**Get_File 测试**:
- `11_getfile_initial.png` - 初始状态
- `12_getfile_pdf.png` - 获取PDF文件
- `13_getfile_image.png` - 获取图片

## 📊 测试验证清单

### Scroll 动作
- [ ] 页面能向下滚动
- [ ] 页面能向上滚动
- [ ] 页面能向左滚动
- [ ] 页面能向右滚动
- [ ] 能在特定坐标位置滚动
- [ ] 滚动距离准确

### Find 动作
- [ ] 能找到可见文本
- [ ] 能找到隐藏文本（自动滚动）
- [ ] 中文文本搜索正常
- [ ] 英文文本搜索正常
- [ ] 文本被正确高亮
- [ ] 高亮持续约2秒

### Get_File 动作
- [ ] 能识别 PDF 文件 (type: 'pdf')
- [ ] 能识别 Word 文档 (type: 'docx')
- [ ] 能识别 Excel 表格 (type: 'xlsx')
- [ ] 能识别图片 (type: 'image')
- [ ] 能识别压缩文件 (type: 'zip', 'rar')
- [ ] 返回正确的文件 URL

## 🔧 故障排查

### 问题: 浏览器测试无法启动
**解决方案**:
- 检查是否有可用的浏览器端点
- 确认 `BROWSER_OUTPUT_PATH` 环境变量设置正确
- 查看配置文件是否存在

### 问题: 截图保存失败
**解决方案**:
- 检查 `screenshots/` 目录权限
- 确认磁盘空间充足
- 查看日志中的错误信息

### 问题: 动作执行失败
**解决方案**:
- 检查坐标是否在有效范围内
- 确认页面已完全加载
- 查看浏览器控制台是否有JavaScript错误

## 📝 自定义测试

你可以创建自己的测试页面：

```html
<!DOCTYPE html>
<html>
<head>
    <title>我的测试页面</title>
</head>
<body>
    <!-- 添加你的测试内容 -->
</body>
</html>
```

然后在 `tool_env.py` 中添加测试代码：

```python
test_page = f"file://{test_pages_dir}/my_test.html"
options = {'start_url': test_page}
# ... 执行测试动作
```

## 📚 相关文档

- [TEST_README.md](../TEST_README.md) - 动作详细说明
- [actions.py](../actions.py) - 动作实现代码
- [action_space_json.py](../action_space_json.py) - 动作定义

## ✨ 贡献

欢迎添加更多测试用例！请确保：
1. 测试页面简洁明了
2. 包含清晰的测试说明
3. 标注建议的测试坐标
4. 更新本README文档
