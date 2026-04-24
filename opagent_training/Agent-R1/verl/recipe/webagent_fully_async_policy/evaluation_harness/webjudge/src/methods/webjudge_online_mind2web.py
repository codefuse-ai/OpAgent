from src.utils import encode_image
from openai import OpenAI
from src.qwen_vlm_request import send_chat_completion_request_shangshu
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

MAX_IMAGE = 15


def openAi(model, messages):
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8030/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return chat_response

def openAi_api(model, messages, temperature=0.6):
    # Set OpenAI's API key and API base to use vLLM's API server.
    chat_response = send_chat_completion_request_shangshu(messages, model, temperature=temperature)

    return chat_response


async def identify_key_points(task, model):
    system_msg = """你是资深Web任务分析专家。你的工作是分析一项给定的任务，找出任务描述中明确提及的关键点（Keypoints）。

**目标**：仔细分析任务描述，并提取任务中明确提到的、有助于实现其目标的Keypoints。

**说明**：
1. 仔细阅读任务描述（如果是英文，请先翻译成中文），进行意图分析。
2. 根据**核心意图**，识别并提取任务描述中直接提及的Keypoints
- **关键点**是指任务描述中明确提及的关键要素、条件或步骤。
- 请勿推断或添加任何未提及的要素、条件、步骤、目的。
- 警惕断章取义造成的意图歪曲

**示例**：
输入：随着与疫情相关的旅行限制持续放宽，哪些航空股可能受益最多？
输出：
1. 航空公司股票
2. 可能受益最多的
3. 疫情相关的旅行限制持续放宽

**回答**：
- **Keypoints**：以编号形式列出完成此任务的明确要点，每行一个，无需解释或其他详细信息。"""

    prompt = """任务: {task}"""
    text = prompt.format(task=task)
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ],
        }
    ]

    openai_res = openAi_api(model, messages, temperature=0)
    response = openai_res.choices[0].message.content

    # responses = await asyncio.to_thread(model.generate, messages)
    return response


async def judge_image(task, image, key_points, model):
    return """1. **Reasoning**: -
2. **Score**: 3"""
    system_msg = """你是一个经验丰富的Web UI元素解读专家 和 WebAgent性能评估专家。你的工作是评估当前Web页面截图是否包含完成Query所需步骤的信息。你的评估必须基于现实世界的网站交互逻辑和常识。
    
**必须遵循的公理**
你的评估必须基于现实世界的网站交互逻辑和常识，而不是理论上的、无休止的怀疑。请遵循以下不可违背的评估公理：”

公理一：【UI状态即声明 (UI State as Declaration)】
教导：“例如：一个被激活（如高亮）的排序选项，是对整个列表状态的权威声明。”
指令：“禁止在识别出有效状态后，再去质疑其有效性。”

公理二：【位置即信息 (Position as Information)】
教导：“例如：在一个已排序的列表中，顶部位置本身就是最强的‘最优’证据。”
指令：“必须将排序列表的顶部项目，直接采纳为任务所要求的最优解。”

公理三：【功能信任原则 (Trust in Functionality)】
教导：“你必须信任网站提供的核心功能（如排序、筛选）是有效的。”
指令：“禁止在没有反证的情况下，对网站排序功能的全局有效性提出质疑”

公理四：【任务边界原则 (Task Boundary Principle)】
教导：“当页面提供了足够支持用户做出合理决策的信息时，任务即告完成。”
指令：“禁止要求进行不必要的深度数据验证。”

**目标**：分析提供的图片，结合常识推理判断其是否包含完成Query所需的必要步骤或信息。在评分之前，请用你的推理来解释你的判断。

**说明**：
1. 【关注UI元素】认真分析Query真实意图，重点关注**与关键点相关**的UI元素（如筛选排序UI、按钮、输入框、高亮、表头等）及其状态
2. 【图片描述】结合**重点关注的UI元素**，提供图片的详细描述，包括其内容、可见元素、文本（如有）以及任何显著特征。
3. 仔细检查图片，评估其是否包含对完成Query至关重要的必要步骤或信息：
- 确定可能与完成Query相关的关键点，例如操作、进度指示器、工具使用方法、已应用的过滤器或分步说明。
- 图片是否显示了与完成Query直接相关的操作、进度指示器或关键信息？
- 这些信息对于理解或确保Query成功是否必不可少？
- 如果图片包含部分但相关的信息，请考虑其实用性，而不是直接忽略它。
4. 请按以下格式提供您的答案：
- **推理**：阐明Query要求。列出你认为需要关注的UI元素及其状态（重点强调最值的状态）。结合UI元素及其状态解释您的思维过程和观察结果。
- **分数**：根据推理结果，使用以下等级进行评分：
- **1**：图片不包含任何必要步骤或相关信息。
- **2**：图片包含的信息极少或模糊，不太可能是必需的。
- **3**：图片包含一些相关步骤或提示，但不够清晰或完整。
- **4**：图片包含重要的步骤或信息，这些步骤或信息高度相关，但并不全面。
- **5**：图片清晰地显示了完成Query所需的必要步骤或信息。

**输出格式**:  
1. **Reasoning**: [你的结合常识的推理过程]  
2. **Score**: [1-5]"""

    jpg_base64_str = encode_image(image, max_size=720)

    prompt = """**任务(Query)**: {task}

**完成Query的关键点*: {key_points}

图中显示的是网页的截图。"""
    text = prompt.format(task=task, key_points=key_points)

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}", "detail": "high"},
                },
            ],
        }
    ]

    # responses = await asyncio.to_thread(model.generate, messages)
    openai_res = openAi_api(model, messages, temperature=0)
    response = openai_res.choices[0].message.content
    return response

async def WebJudge_Online_Mind2Web_eval(task, last_actions, images, model, score_threshold, final_result_response, thoughts):
    system_msg = """# 角色定义
你是一个经验丰富的Web UI元素解读专家、WebAgent性能评估专家。具备以下核心能力：
1. 通过Web截图中UI元素（如搜索框、筛选框）状态（如高亮、下划线），理解WebAgent的操作轨迹能力
2. 多维评估WebAgent的操作轨迹质量
3. 基于视觉证据的严格验证能力

## WebAgent设计初衷原则
WebAgent的核心使命是**端到端完成用户任务**，在未达到必须人工介入的临界点前，不得将操作步骤交还用户。评估时需优先验证：
1. 是否正确理解用户意图
2. 是否已完成所有的关键点Key Points
3. 没有过早终止任务流程
4. 是否规避了本应自动完成的交互环节

# 任务描述
根据用户的Query、Query完成的Key Points、WebAgent的操作历史记录、操作轨迹中一些连续的网页截图（Trajectory screenshot），你的目标是按照评估框架，为WebAgent的这次操作表现评估打分。

## 操作代表的含义
click:Click on an element with coordinates on the screenshot of the webpage.
type:Type content into a field with a specific id
hover:Hover over an element with the coordinates
press:Simulate pressing a key or a key combination
scroll:Scroll the page up or down
hscroll:Scroll the page horizontally
new_tab:Open a new, empty browser tab
tab_focus:Switch browser focus to a specific tab
close_tab:Close the currently active browser tab
goto:Navigate to a specific URL
go_back:Navigate to the previously viewed page
go_forward:Navigate to the next page after a 'go_back' action
move_to:Move the cursor to a specific location without clicking
double_click:Perform a double click at a specific location
browser_select_option:Select an option from a dropdown menu
wait:Wait for the change to happen

# 评估框架
一切以截图为证据，采用六维评估体系

## 评估维度与标准

### 核心维度

#### 1. 完成度评分（Completion Score 6级）
视觉唯一依据：一切以截图为证据，结合常识推理判断其是否包含完成Query所需的必要步骤或信息。
最终状态验证：只关心最终状态是否达成了用户的核心目标，它是一个结果导向的评分。

*重要评估标准*
1.  **筛选结果必须被正确展示。** 如果筛选器未能被妥善应用（例如：缺少选择、未点击确认，或结果中无任何可见效果），则该任务不被视为成功。

2.  **你必须仔细检查截图和行为历史是否满足这些关键点。** 确保诸如“最佳”、“最高”、“最便宜”、“最新”、“最近”、“最低”、“最近”、“评分最高”、“最大”和“最新”等特定的筛选条件，都已通过筛选器被正确应用。

3.  **特定的关键点或要求应通过筛选器来应用。** 否则，将所有要求作为输入文本进行搜索将被视为失败，因为它无法保证所有结果都满足要求！

4.  **如果任务要求按特定的金额、年份、或卧室/浴室数量范围进行筛选，所应用的筛选器必须与给定的要求完全匹配。任何偏差都将导致失败。** 为确保任务成功，所应用的筛选器必须精确匹配指定的范围，既不能过宽也不能过窄。
    
    **失败案例举例：**
    - 如果要求是低于$50，但应用的筛选器是低于$25，则为失败。
    - 如果要求是$1500-$2500，但应用的筛选器是$2000-$2500，则为失败。
    - 如果要求是$25-$200，但应用的筛选器是$0-$200，则为失败。
    - 如果要求的年份是2004-2012，但应用的筛选器是2001-2012，则为失败。
    - 如果要求的年份是2015年以前，但应用的筛选器是2000-2014，则为失败。
    - 如果任务要求正好是2间卧室，但应用的筛选器是2+间卧室，则为失败。

5.  某些任务需要一个**提交动作**或**结果的展示**才被视为成功。

6: **是否存在判断型任务**：检查路径的完备性。如否 视为不满足。豁免条款：完备检查核心页面后，依然没有找到任何信息，构成否定性证据，Query仍应被视为完成。

7: **筛选器与搜索框合理结合使用**：检查是否符合一个经验丰富用户的操作逻辑（错误案例： 在购物网站搜索框输入“尿不湿 销量最高的”。正确做法：搜索框输入“尿不湿”并用筛选器按照销量排序）

| 等级 | 标准要求 |
|------|----------|
| -1 | 遇到拦截导致无法达成（最高优先级）: 在WEB上使用功能（如搜索、登录、收藏、评论等）时，遇到网络问题、需要登录、安全拦截、人机验证等拦截导致任务失败 |
| 5 | 完美达成:完全满足WebAgent设计初衷原则和**重要验证准则** |
| 4 | 核心达成:满足核心目标但存在非关键参数或信息偏差（如价格过滤偏差），或最终状态距离端到端完成任务只差一步（如未进入详情页面），或在网站限制下的“最接近完美”的达成 |
| 3 | 部分达成:执行方向性正确、完成主要步骤，但缺失关键操作节点或不满足1-2个**重要验证准则** |
| 2 | 方向错误:执行方向性错误 或 不满足多个**重要验证准则**，但部分操作有效  |
| 1 | 完全失败:完全没有执行有效操作或完全偏离任务目标 |

#### 2. 动作有效性评分（Action Score 5级）
请根据每步的截图变化，结合思考过程，评估操作的准确性。
焦点： 评估每一步微观操作的“意图-执行-反馈（根据截图变化判断操作有效性）”闭环是否成功。

| 等级 | 标准要求 |
|------|----------|
| 5 | 精准无误：所有操作都精准命中目标元素，参数正确，并产生了预期的界面反馈。 |
| 4 | 轻微偏差：绝大多数操作精准，但存在1-2次对非关键元素的误操作（很快被修正）或非关键参数的微小偏差。 |
| 3 | 偶有失效：存在少量需要多次尝试才能成功的操作，或在无歧义的场景下出现了目标元素的偏离 |
| 2 | 频繁失效：存在大量未命中目标的操作，或对关键元素/参数的错误操作（如选错日期、点错提交按钮）且未及时修正 |
| 1 | 完全无效：绝大多数或所有操作均未命中目标，或未产生任何有效的界面反馈 |

#### 3. 轨迹效率评分（Trajectory Score 5级）
焦点： 评估整个行为过程是否符合一个经验丰富用户的操作逻辑，即路径是否合理、经济

| 等级 | 标准要求 |
|------|----------|
| 5 | 最优路径（理论最小步骤数）且 符合一个经验丰富用户的操作逻辑 |
| 4 | 可接受冗余（≤理论步骤+2）或 行为过程略显生疏 |
| 3 | 明显冗余（理论×1.5倍） |
| 2 | 路径存在循环/反复 或 行为过程违背常识 |
| 1 | 完全无逻辑的操作序列 或 任务失败且路径过短未进行充分尝试 |

# 任务流程
1. **目标解析**：分解Query的3级子目标
2. **维度评估**：独立评定各维度等级
3. **综合判定**：输出结构化评估结果

# 执行保障
1. **视觉证据优先**：所有判断必须基于截图的可验证信息

# 输出规范
<Thoughts>:
1.结合目标解析进行维度评估：
完成度(Completion Score)：[推理过程：按照结果导向完成度评分标准评估。得分：-1~5]
动作有效性评分(Action Score)：[根据每步的截图变化反推操作元素定位正确性：并按照动作有效性评分标准要求评估。得分：1~5]
轨迹效率(Trajectory Score)：[推理过程：轨迹效率评分标准要求。得分：1~5]
<Thoughts>
```json
{
"Completion Score": <完成度评分-1~5>,
"Action Score": <动作有效性评分1~5>,
"Trajectory Score": <轨迹效率评分1~5>
}
```

"""
    prompt = """
    请严格遵守以下规则：
    1.  **输入的图片是唯一的事实依据 (Ground Truth)**：连续截图中的视觉内容变化，是评分的唯一依据。
    2.  **输入的文本是待验证的声明 (Claim)**：描述了期望发生的事情，但它本身可能错误或未成功，你需要去验证它。
    
    用户 任务(Query): {task}

    Key Points: {key_points}

    WebAgent的操作历史轨迹: {last_actions}

    按照操作前后顺序排列的Trajectory screenshot:
"""

    key_points = await identify_key_points(task, model)
    key_points = key_points.replace("\n\n", "\n")
    logger.info(f"key_points:{key_points}")

    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

    # tasks = [judge_image(task, image, key_points, model) for image in images]
    # image_responses = await asyncio.gather(*tasks)

    whole_content_img = []
    whole_thoughts = []
    record = []

    for image in images:
        jpg_base64_str = encode_image(image, max_size=1024)
        whole_content_img.append(
            {
                'type': 'image_url',
                'image_url': {"url": f"data:image/png;base64,{jpg_base64_str}", "detail": "high"}
            }
        )

    whole_content_img = whole_content_img[-MAX_IMAGE:]
    if len(whole_content_img) == 0:
        prompt = """请严格遵守以下规则：
        1.  **输入的图片是唯一的事实依据 (Ground Truth)**：连续截图中的视觉内容变化，是评分的唯一依据。
        2.  **输入的文本是待验证的声明 (Claim)**：描述了期望发生的事情，但它本身可能错误或未成功，你需要去验证它。

        User Task: {task}

Key Points: {key_points}

WebAgent的操作历史轨迹:
{last_actions}
按照操作前后顺序排列的Trajectory screenshot:
"""
    text = prompt.format(task=task,
                         last_actions="\n".join(f"{i + 1}. {action}" for i, action in enumerate(last_actions)),
                         key_points=key_points)

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                           {"type": "text", "text": text}]
                       + whole_content_img
        }
    ]
    return messages, text, system_msg, record, key_points