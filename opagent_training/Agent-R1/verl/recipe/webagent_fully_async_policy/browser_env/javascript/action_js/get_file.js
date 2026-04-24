(args) => {
    const [x, y] = args;
    // 获取(x, y)处的所有堆叠元素
    const elements = document.elementsFromPoint(x, y);

    if (!elements || elements.length === 0) {
        return null;
    }

    /**
     * 辅助函数：从URL中提取文件类型（扩展名）。
     * @param {string} url - 待检查的URL。
     * @returns {string|null} - 如果匹配成功，返回文件扩展名的小写字符串 (如 'pdf', 'docx')，否则返回 null。
     */
    const getFileTypeFromUrl = (url) => {
        if (!url) return null;

        const fileExtensionsRegex = /\.(pdf|docx?|xlsx?|pptx?|zip|rar|tar|gz|csv|txt|epub|mobi|mp4|mov)\b/i;

        try {
            const path = new URL(url, window.location.href).pathname;
            const match = fileExtensionsRegex.exec(path);

            // 如果匹配成功，match[1] 将是括号内捕获的扩展名
            if (match && match[1]) {
                return match[1].toLowerCase();
            }
            return null;
        } catch (e) {
            return null; // 无效的URL
        }
    };

    // 遍历所有堆叠的元素，从最顶层的元素开始
    for (const element of elements) {
        // Case 0: 优先查找文件链接 ---
        // 查找当前元素或其最近的父级<a>标签
        const anchor = element.closest('a');
        if (anchor && anchor.href) {
            // 调用辅助函数获取具体文件类型
            const fileType = getFileTypeFromUrl(anchor.href);
            if (fileType) {
                // 将获取到的文件类型（如'pdf'）放入type字段
                return {
                    type: fileType,
                    url: new URL(anchor.href, window.location.href).href
                };
            }
        }

        // 查找图片逻辑
        // Case 1: 元素本身就是 <img>
        if (element.tagName === 'IMG' && element.src) {
            return { type: 'image', url: element.src };
        }

        // Case 2: 从当前元素向下查找第一个 <img> (这个逻辑可以被Case 1覆盖，但保留以防万一)
        const img = element.querySelector('img');
        if (img && img.src) {
            return { type: 'image', url: img.src };
        }

        // Case 3: 检查背景图片 background-image
        const style = window.getComputedStyle(element);
        const bgImage = style.backgroundImage;
        if (bgImage && bgImage !== 'none') {
            const urlMatch = bgImage.match(/url\("?([^"]+)"?\)/);
            if (urlMatch && urlMatch[1]) {
                // 返回绝对URL
                return { type: 'image', url: new URL(urlMatch[1], window.location.href).href };
            }
        }
    }

    // 原始代码中的向上遍历逻辑在这里可以被省略，因为 `document.elementsFromPoint`
    // 已经提供了从上到下的元素堆栈，`element.closest('a')` 已经覆盖了向上查找链接的场景。
    // 保留原始的向上查找图片逻辑作为最后的备用方案。
    let parent = elements[0].parentElement;
    while(parent) {
        const parentImg = parent.querySelector('img');
        if (parentImg && parentImg.src) {
            return { type: 'image', url: parentImg.src };
        }
        const parentStyle = window.getComputedStyle(parent);
        const parentBgImage = parentStyle.backgroundImage;
        if (parentBgImage && parentBgImage !== 'none') {
             const urlMatch = parentBgImage.match(/url\("?([^"]+)"?\)/);
            if (urlMatch && urlMatch[1]) {
                return { type: 'image', url: new URL(urlMatch[1], window.location.href).href };
            }
        }
        parent = parent.parentElement;
    }

    return null; // 彻底找不到
}
