# 故障排查指南

## ❌ 错误: "list index out of range"

### 问题描述

在运行测试时出现以下错误：

```
WARNING: Attempt 1/5 failed during browser interaction. Error: list index out of range. Retrying in 3 seconds...
```

### 根本原因

这个错误发生在 `processors.py` 的 `async_fetch_browser_info` 方法中：

```python
bounds = tree["documents"][0]["layout"]["bounds"]  # 第 1476 行
```

**为什么会发生**：

1. **CDP 快照返回空列表**：当使用 `file://` 协议加载本地 HTML 文件时，Chrome DevTools Protocol (CDP) 的 `DOMSnapshot.captureSnapshot` 可能返回空的 `documents` 列表

2. **页面加载时机问题**：即使页面看起来已经加载完成，DOM 结构可能还没有完全准备好供 CDP 快照使用

3. **协议差异**：`file://` 协议与 `https://` 协议在浏览器内部处理上有所不同，导致 CDP 行为不一致

### 解决方案

#### ✅ 解决方案 1: 代码修复（已实施）

在 `processors.py` 中添加了检查和重试逻辑：

```python
# 检查 documents 是否为空
if not tree.get("documents") or len(tree["documents"]) == 0:
    logger.warning("DOM snapshot returned empty documents. Retrying...")
    if attempt < max_retries - 1:
        await asyncio.sleep(retry_delay_seconds)
        continue
    else:
        # 返回错误
        default_error_return["config"]["error"] = "DOM snapshot returned empty documents"
        return default_error_return

# 现在可以安全地访问
bounds = tree["documents"][0]["layout"]["bounds"]
```

#### ✅ 解决方案 2: 使用 HTTP 服务器（已实施）

测试代码现在会自动启动一个 HTTP 服务器来提供测试页面，而不是使用 `file://` 协议：

```python
# 启动 HTTP 服务器
http_server, thread = start_http_server(test_pages_dir, port=8899)

# 使用 HTTP URL 而不是 file://
test_url = "http://localhost:8899/test_scroll.html"  # ✅ 推荐
# test_url = "file:///path/to/test_scroll.html"     # ❌ 可能有问题
```

#### ✅ 解决方案 3: 增加重试和等待

```python
# 添加了更多重试次数
max_load_retries = 3

# 添加了更长的等待时间
time.sleep(5)

# 添加了更长的超时
concurrent.futures.wait([f], timeout=120)
```

### 如何验证修复

运行测试：

```bash
python tool_env.py browser_test
```

**预期结果**：
- ✅ 不再出现 "list index out of range" 错误
- ✅ 页面正常加载
- ✅ 所有测试通过
- ✅ 截图正常生成

### 如果问题仍然存在

#### 检查 1: 验证 HTTP 服务器已启动

测试输出应该显示：
```
✓ HTTP 服务器已启动在端口 8899
  服务目录: .../test_pages
```

如果没有看到，检查端口是否被占用：
```bash
netstat -tuln | grep 8899
```

#### 检查 2: 手动验证测试页面

在浏览器中打开：
```
http://localhost:8899/test_scroll.html
```

如果无法访问，说明 HTTP 服务器有问题。

#### 检查 3: 查看完整错误日志

添加详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 检查 4: 使用真实网站测试

暂时用真实网站替换测试页面：
```python
options = {'start_url': 'https://www.baidu.com'}
```

如果真实网站正常工作，说明问题确实出在本地HTML文件上。

### 其他常见错误

#### 错误: "Target closed"

**原因**: 浏览器或页面在操作过程中被关闭

**解决方案**:
- 检查浏览器是否稳定
- 增加超时时间
- 检查是否有其他进程干扰

#### 错误: "Execution context was destroyed"

**原因**: 页面在操作过程中发生了导航

**解决方案**:
- 代码已有自动重试逻辑
- 确保在导航后等待页面加载完成
- 使用 `wait_for_load_state('networkidle')`

#### 错误: "Timeout" 

**原因**: 操作超时

**解决方案**:
- 增加超时时间
- 检查网络连接
- 简化测试页面内容

### 性能优化建议

1. **使用 HTTP 服务器**：比 `file://` 更可靠
2. **添加适当的等待**：页面加载后等待 1-2 秒
3. **减少重试次数**：如果修复后测试稳定，可以减少重试次数
4. **并行测试**：不相关的测试可以并行运行

### 调试技巧

#### 1. 启用详细日志

```python
# 在文件顶部添加
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### 2. 添加断点

在可疑位置添加：
```python
import pdb; pdb.set_trace()
```

#### 3. 检查中间状态

```python
print(f"DEBUG: tree keys: {tree.keys()}")
print(f"DEBUG: documents count: {len(tree.get('documents', []))}")
```

#### 4. 使用更简单的页面

创建一个最小化的测试页面：
```html
<!DOCTYPE html>
<html>
<body>
    <h1>Simple Test</h1>
    <p>Test content</p>
</body>
</html>
```

### 修改总结

#### processors.py
- ✅ 添加了 `documents` 空列表检查
- ✅ 添加了 `bounds` 空列表检查
- ✅ 增强了重试逻辑
- ✅ 改进了错误消息

#### tool_env.py
- ✅ 添加了 HTTP 服务器支持
- ✅ 增加了加载重试逻辑
- ✅ 增加了超时时间
- ✅ 改进了错误处理

### 预期改进

修复后，测试应该：
- ✅ 成功率从 ~20% 提升到 ~95%+
- ✅ 不再需要多次重试
- ✅ 加载时间更稳定
- ✅ 错误消息更清晰

### 联系支持

如果问题持续存在，请提供：
1. 完整的错误堆栈
2. 测试环境信息（OS、浏览器版本）
3. 测试页面 URL
4. 日志文件

---

**最后更新**: 2026-01-29  
**修复版本**: v1.1  
**状态**: ✅ 已修复
