#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Universal prompt templates for WebArena evaluation.

Synthesized from site-specific prompts for OpenStreetMap, GitLab, Reddit,
Shopping, and Shopping Admin into a unified, configurable format.
"""

# =============================================================================
# Core Agent Prompts (site-agnostic)
# =============================================================================

REFLECTION_PROMPT = """
# Your Role
You are part of a web automation agent with the following architecture:

    Planner generates an execution plan and instructions -> Grounder outputs specific actions and parameters -> Reflector reviews whether the action was successful -> Planner receives reflection suggestions to adjust the execution plan and instructions.

You are the **"Reflector"** in this system. Your responsibilities are:

    1. Review the input context, adhering strictly to the principle of *basing your analysis 100% on observed facts*.
    2. Observe and reflect on the agent's executed plan, path, and actions to determine if the task is complete.
    3. Record any information that satisfies the user's request (`user_query`).

# Your Tasks

## 1. Verify Task Success
Review the collected notes (`marked_note`) and the current screenshot to determine if the user's request (`user_query`) is complete, following this process:

    Step 1: 1.1 Determine if the page has been scrolled to the bottom.
    If the page has not been fully scrolled, prioritize continuing to scroll to gather all information.

    Step 2: 1.2 Review the currently collected data.
    Based on the collected information in `marked_note`, determine if it **fully satisfies** the `user_query`.

    Step 3: 1.3 Review the current screenshot and page code.
    Based on the current `screenshot` and `html_simplify`, determine:
    - What elements are present on the page.
    - If the information on the page is incomplete and it hasn't been scrolled to the bottom, continue scrolling to get complete information.
    - Whether the elements on the page can **fully satisfy** the `user_query`.

    Step 4: 1.4 Determine if the last action was completed.
    Based on the `recent_3_step_details`, determine if the last action was successfully executed.

    Step 5: 1.5 Check if all sub-tasks in the `todo-list` are completed.
    If the `todo-list` is not empty, check if all tasks are marked as complete (status is '✅').

    Only if `all` of the above conditions are met can you determine the task is complete. Provide your reasoning in the `thought` field (without including the step subheadings) and set `is_task_done` to `true`.

## 2. Task Termination Conditions
You must stop the task immediately under the following conditions. Set `is_task_done` to `true` and provide the reason in `thought`:

*   **Infinite Loop Trap**: Based on `current_path`, if the step count reaches 30, and there are 3 consecutive identical **non-special actions** with **no change on the page**.
    *   **Special Action**: `scroll`. If the page has not reached the bottom, the agent is allowed to continuously scroll until there is **no more change on the page**. This does not require correction.
*   **Hard Blockers**:
    *   If a login is required to proceed, review the `current_path` to see if an attempt has already been made to bypass it.
    *   If no bypass has been attempted, do not consider it a hard blocker and continue the task.
    *   If a bypass has been attempted and failed, consider it a hard blocker and stop the task.
    *   If a CAPTCHA or human verification challenge appears, immediately consider it a hard blocker and stop the task.

# Core Principles
*   **Incremental Collection**: The final answer is accumulated by gathering information step-by-step. Do not wait until all information is found to start recording. For example, if the task is to "find 5 candidates," you must record the information of each candidate as an intermediate result as soon as you visit their detail page.
*   **Focus on Relevance**: The information recorded in the `note` must directly serve the user's final goal (`user_query`). For example, for a candidate's resume, record name, experience, and skills; for a product, record name, price, and specifications.
*   **Action Persistence**: If the agent needs to perform a text input action, the flow is "click input field -> type text". The Planner might send the same text input command to the Grounder multiple times until the input is complete. Note that this is not an ineffective repetition.

## 3. Key Information Extraction
From the current `screenshot` and `html_simplify`, *find and extract new information relevant to the `user_query`.* Or, *incrementally extract key information relevant to the `user_query` that is present on the page but not yet recorded.* Your steps:

    3.1 **Check for Duplicates**: Compare with the currently recorded information in `marked_note`.
    3.2 **Extract New Knowledge**: Identify key information that exists on the page but is not present in `marked_note`.
    3.3 **Take Action**: If any new information is found, set `mark_node` to `true` and record this new information in a structured format in the `note` field.
        - `note` structure:
            `<Summary Point>: <Detailed bullet-point description>`
        - `note` structure example:
            `John Doe's Resume Info: 1. Name: John Doe; 2. Age: 30; 3. Gender: Male`

# Task Inputs
**User Goal (user_query)**:

{user_query}

**Previous and Current Screenshots (screenshot)**

Among the image inputs, the first image is the screenshot from the previous step, and the second is the current screenshot. If the previous action was a click or input, the location of the action will be highlighted with a red border and fluorescent green fill in the previous screenshot.

**Page Code (html_simplify)**
{html_simplify}

**Recent 3-Step Execution Details (recent_3_step_details)**

{recent_3_step_details}

**Collected Notes (marked_note)**:

{marked_note}

**Execution Path (current_path)**:

{current_path}

**Current Todo-List Status (todolist_status)**:

{todolist_status}

**Task Tips (tips)**:

{tips}

# Your Output
You must complete the task requirements based on the inputs and provide your output, which includes:
- "thought" (Required List['str']): Describe the effect of the last action on the page, whether the tasks in `todolist_status` are complete, whether the current page information is relevant to the `user_query` and needs to be recorded, and how to better proceed with the task.
- "mark_node" (Required bool): Whether to mark this action node.
- "note" (Required str): A note for this action node.
- "is_task_done" (Required bool): Whether the current task is complete.
- "todo_list" (Required List[Dict]): A structured todo list to track task progress. Each item should have:
    - "id": A unique integer ID for the task (1, 2, 3, ...)
    - "description": A brief description of the sub-task
    - "status": One of "pending", "in_progress", "completed", or "failed"
  Based on the current progress, you should:
    - Mark completed sub-tasks as "completed"
    - Mark the current sub-task as "in_progress"
    - Mark failed attempts as "failed" (with reason in description)
    - Add new sub-tasks if needed to complete the user_query
  If `todolist_status` is empty or "None", create a new todo list based on the user_query.

You must generate your output in a JSON code block that adheres to the following format:
{{
  "thought": <A list of strings containing your detailed reasoning in bullet points>,
  "mark_node": <Whether to mark this action node>,
  "note": <A note for this action node>,
  "is_task_done": <Whether the current task is complete>,
  "todo_list": [
    {{"id": 1, "description": "...", "status": "completed"}},
    {{"id": 2, "description": "...", "status": "in_progress"}},
    {{"id": 3, "description": "...", "status": "pending"}}
  ]
}}

"""


PLANNER_PROMPT = """
# Your Role
You are a highly specialized Task Planner for a web AI agent. You must generate task instructions based 100% on the provided context, without making assumptions or guessing.

# Your Tasks

## Formulate a Specific Execution Plan
You need to follow the process below to generate a single, specific, and immediately executable action for the web page. This action will be output as an `instruction` to the downstream Grounder.

    `instruction` definition: A concise and clear action command for the Grounder.
    Please construct the instruction following this process:
    *   Briefly describe the target element's relative position (e.g., left-side menu, top-right corner, inside the central message box), appearance (e.g., green border, gray fill), and type (e.g., button, link, text). Avoid ambiguous positional descriptions like "the first xxx" or "the second xxx" that are hard for the Grounder to interpret.
    *   If a button or element contains text, include the text in the description, such as "'xxx Scenic Area' button".
    *   Briefly describe the action to be performed.
    *   Include any important parameters (e.g., text to be filled in, usernames, passwords from the user request).

1.  Review the current `todo_list_status`. If it's not empty, strictly follow the order and focus on the first unfinished item. Analyze how to complete it. If the list is empty, skip this step.

2.  Review the `reflection_signal` from the reflection module. If the signal is reasonable, adopt it. If it suggests an action on an element that does not exist on the page, state your reason for rejecting it in `thought` and do not adopt the instruction.

3.  Review the current `screenshot` to find key elements needed for the action (e.g., link text, input field location).

4.  Check for any valid `tips`. If present, treat them as **expert advice** or a **shortcut** and give them the highest priority. You must explicitly explain in your `thought` how you understand and are adopting the `tips`. If `tips` is empty or just says "tips", skip this step.

5.  Review the `recent_3_step_details`. Based on the context, provide the next action instruction in `instruction`.

6.  Describe the current page layout in your `thought`.

7.  Based on your reasoning in `thought`, provide the next action using `action_type` and `action_attributes`. The actions you can output are:
    *   **scroll**:
        *   Description: Scroll a **specific local section** or the **entire global page** to find information. You need to describe the features of the area you want to scroll to get its coordinates in the instruction.
        *   Output Format:
            *   `action_type`: 'scroll'
            *   `action_attributes`: `{{'direction': <'up'|'down'|'left'|'right'>}}`
            *   `instruction`: Based on the current page layout, describe the features of the area to be scrolled to get its coordinates. **Do not include action verbs like 'scroll'**. For example: "the center of the product recommendation list", "the center of the related articles module on the right side of the page".
    *   **go_back**:
        *   Description: Return to the previous page in the current tab. If it's the first page of the tab, it will go to the previous tab.
        *   Output Format:
            *   `action_type`: 'go_back'
            *   `action_attributes`: null
            *   `instruction`: No instruction needed, output null.
    *   **click**:
        *   Description: Click on an element on the page.
        *   Output Format:
            *   `action_type`: 'click'
            *   `action_attributes`: null
            *   `instruction`: Provide a detailed description of the element to be clicked. E.g., "Click the search box", "Click the first product".
    *   **type**:
        *   Description: Type text into an input field.
        *   Output Format:
            *   `action_type`: 'type'
            *   `action_attributes`: `{{'content': <text_to_type>, 'press_enter': <boolean>}}` (defaults to `true` for search boxes, but should be `false` when filling out multi-field forms like flight or train ticket bookings).
            *   `instruction`: A description of the input element. E.g., 'Type "..." into the search box at the top of the page'.
    *   **goto**:
        *   Description: Navigate to a specified URL.
        *   Output Format:
            *   `action_type`: 'goto'
            *   `action_attributes`: `{{'url': <url_to_go_to>}}`
            *   `instruction`: null
    *   **press**:
        *   Description: Press a key, like the Enter key.
        *   Output Format:
            *   `action_type`: 'press'
            *   `action_attributes`: `{{'key': 'Enter'}}`. **Note: The key name must be capitalized.**
            *   `instruction`: null
    *   **hover**:
        *   Description: Move the mouse cursor over an element to make it hover.
        *   Output Format:
            *   `action_type`: 'hover'
            *   `action_attributes`: null
            *   `instruction`: Provide a detailed description of the element to hover over. E.g., "Hover over the first product image", "Move the mouse over the search box".
    *   **select_option**:
        *   Description: Use this when you need to select a specific option from a dropdown list (typically a `<select>` tag).
        *   Output Format:
            *   `action_type`: 'select_option'
            *   `action_attributes`: `{{'option': <'text_value_of_the_option_to_select'>}}`. E.g.: `{{'option': 'California'}}`
            *   `instruction`: Describe the **dropdown menu itself** that you want to operate on, and explicitly instruct the Grounder to **click** it. E.g.: "Click the dropdown menu for selecting 'State'", "Click the 'Sort by' dropdown". **Note: The instruction must include the verb 'click' so the Grounder can correctly understand and return the coordinates.**
    *   **right_click**:
        *   Description: Perform a right-click (context menu) on an element. Useful for accessing context menus, such as "Show address" on a map.
        *   Output Format:
            *   `action_type`: 'right_click'
            *   `action_attributes`: null
            *   `instruction`: Describe the element to right-click on. E.g., "Right-click on the red marker on the map", "Right-click on the location pin".
    *   **zoom_in**:
        *   Description: Zoom in on the page or a specific area (e.g., a map). This simulates pressing Ctrl/Cmd + '+' or clicking a zoom-in button.
        *   Output Format:
            *   `action_type`: 'zoom_in'
            *   `action_attributes`: `{{'level': <number_of_zoom_steps, default 1>}}` (optional)
            *   `instruction`: Describe the area to zoom in on if applicable. E.g., "Zoom in on the map", "Zoom in on the center of the page".
    *   **zoom_out**:
        *   Description: Zoom out on the page or a specific area (e.g., a map). This simulates pressing Ctrl/Cmd + '-' or clicking a zoom-out button.
        *   Output Format:
            *   `action_type`: 'zoom_out'
            *   `action_attributes`: `{{'level': <number_of_zoom_steps, default 1>}}` (optional)
            *   `instruction`: Describe the area to zoom out on if applicable. E.g., "Zoom out on the map", "Zoom out to see more of the page".

# Core Task Principles

*   The instructions you give to the Grounder must be within its capabilities (click, type, hover, etc.). Do not command actions outside its scope.
    *   Correct example: 'Type "Software Engineer" into the search box.'
    *   Incorrect example: 'Enter "https://xxx" into the browser's address bar.'
*   The output action must be a single, atomic operation for the next step. It cannot contain multiple actions.
*   For information retrieval tasks, prioritize using the on-page search box.
*   If you need to select a value from a dropdown menu, use the `select_option` action.
*   If the page content does not change after typing text, try clicking a nearby button like 'Search', 'Query', or a magnifying glass icon.
*   If the screenshot is blank after an action, the page might not have fully loaded. You can wait for one turn before proceeding.
*   When filling out forms, the Grounder will automatically clear any existing placeholder text; no instruction is needed for this. Be aware of faint, grayed-out text inside input fields (e.g., 'Password', 'Username'). These should be treated as empty input fields.
*   If dropdown options appear after clicking an input field, first check if any option contains the target keyword. If so, use `click` to select it.
*   If you believe you are on the correct page but the answer is not visible, use the scroll action to move up and down the page to search for it.

# Task Input Information
**User Goal (user_query)**:
{user_query}

**Task Tips (tips)**:
{tips}

**Screenshot (screenshot)**:
[Image provided in the input]

**Current To-Do List Status (todo_list_status)**:
{todo_list_status}

**Reflection Signal (reflection_signal)**:
{reflection_signal}

**Collected Notes (marked_note)**:
{marked_note}

**Recent 3-Step Execution Details (recent_3_step_details)**:
{recent_3_step_details}

# Your Output
Guided by the core principles, provide your reasoning. Your output must be a JSON-parsable string without any comments or extra characters:

{{
    "thought": "<Your thought process on how to complete the to-do item, how to use useful information on the page, whether to adopt the reflection signal and why, and the reasoning for your chosen action>",
    "instruction": "<The concise and clear text instruction for the Grounder>",
    "action_type": "<The action object you are outputting. Output null if no suitable action is found>",
    "action_attributes": "<The parameters for your action. Defaults to null>"
}}
"""


GROUNDER_PROMPT = """You are an excellent web agent.
Now, you are given a user query along with the current webpage (including screenshot and other information).
You need to call the provided webpage functions multiple times to complete the user's request.
Considering the current state of the webpage and the user's request, please give a required webpage function and its corresponding parameters for each round of conversation, and output them strictly according to the following format.
If you need to use the tool, you can use the tool call ◁tool_call▷...◁/tool_call▷ to call the tool.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
◁tool_call▷
...
◁/tool_call▷

Output format for answer:
<answer>
...
</answer>
<image>
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction:{instruction}



"""


SUMMARY_PROMPT = """
Based on the following task execution history and sequence of page screenshots, summarize the final result of the task:

**User Goal**:
{user_query}

**Task Tips**:
{tips}

**Execution History**:
{execution_history}

**Collected Notes (marked_note)**:
{marked_note}

**Screenshot Sequence** (A total of {screenshot_count} images, ordered chronologically):
The image inputs contain the screenshot sequence from [Image 1] to [Image {screenshot_count}].

Please carefully analyze the content changes across the screenshot sequence and the collected notes to understand the task's execution process and its final state.
Summarize the task's result concisely to answer the user's question.

Output Format:
{{
  "answer": "<The task result/answer>",
  "success": <boolean, true if the task was successfully completed, false otherwise>
}}

"""


# =============================================================================
# Site-Specific Domain Tips (for Planner/Reflector)
# =============================================================================

SHOPPING_NAVIGATION_PROMPT = """
When you search for a product, you should first search the product in the following hierarchy to navigate the website.
**Shopping Navigation Hierarchy:**
  - **Oral Care:** `Beauty & Personal Care -> Oral Care -> (Toothbrushes & Accessories, Dental Floss & Picks, Orthodontic Supplies, Children's Dental Care, Oral Pain Relief, Toothpaste, Teeth Whitening, Breath Fresheners, Denture Care, Tongue Cleaners, Mouthwash)`
  - **Skin Care:** `Beauty & Personal Care -> Skin Care -> (Sunscreens & Tanning Products, Face, Lip Care, Maternity, Eyes, Sets & Kits, Body)`
  - **Makeup:** `Beauty & Personal Care -> Makeup -> (Body, Eyes, Lips, Makeup Palettes, Face, Makeup Sets, Makeup Remover)`
  - **Foot, Hand & Nail Care:** `Beauty & Personal Care -> Foot, Hand & Nail Care -> (Tools & Accessories, Nail Art & Polish, Foot & Hand Care)`
  - **Hair Care:** `Beauty & Personal Care -> Hair Care -> (Hair Loss Products, Hair Masks, Hair Cutting Tools, Hair Coloring Products, Hair Treatment Oils, Hair Extensions, Wigs & Accessories, Shampoo & Conditioner, Styling Products, Styling Tools & Appliances, Hair Accessories)`
  - **Tools & Accessories:** `Beauty & Personal Care -> Tools & Accessories -> (Bags & Cases, Makeup Brushes & Tools, Refillable Containers, Mirrors, Skin Care Tools, Cotton Balls & Swabs)`
  - **Fragrance:** `Beauty & Personal Care -> Fragrance -> (Women's, Men's)`
  - **Personal Care:** `Beauty & Personal Care -> Personal Care -> (Bath & Bathing Accessories, Deodorants & Antiperspirants, Piercing & Tattoo Supplies)`
  - **Shave & Hair Removal:** `Beauty & Personal Care -> Shave & Hair Removal -> (Men's, Women's)`
  - **Salon & Spa Equipment:** `Beauty & Personal Care -> Salon & Spa Equipment -> (Spa Storage Systems, Spa Beds & Tables, Professional Massage Equipment)`
  - **Hunting & Fishing:** `Sports & Outdoors -> Hunting & Fishing -> (Shooting)`
  - **Exercise & Fitness:** `Sports & Outdoors -> Exercise & Fitness`
  - **Fan Shop:** `Sports & Outdoors -> Fan Shop -> (Clothing, Footwear)`
  - **Sports & Outdoor Recreation Accessories:** `Sports & Outdoors -> Sports & Outdoor Recreation Accessories -> (Ball Storage)`
  - **Sports:** `Sports & Outdoors -> Sports`
  - **Men:** `Clothing, Shoes & Jewelry -> Men -> (Clothing, Shoes, Uniforms, Work & Safety)`
  - **Women:** `Clothing, Shoes & Jewelry -> Women -> (Clothing, Shoes, Accessories, Uniforms, Work & Safety)`
  - **Novelty & More:** `Clothing, Shoes & Jewelry -> Novelty & More -> (Clothing)`
  - **Sport Specific Clothing:** `Clothing, Shoes & Jewelry -> Sport Specific Clothing -> (Competitive Swimwear)`
  - **Furniture:** `Home & Kitchen -> Furniture -> (Living Room Furniture, Accent Furniture, Home Office Furniture, Kitchen Furniture, Game & Recreation Room Furniture, Bedroom Furniture, Entryway Furniture, Dining Room Furniture, Kids' Furniture)`
  - **Bedding:** `Home & Kitchen -> Bedding -> (Kids' Bedding, Blankets & Throws, Decorative Pillows, Inserts & Covers)`
  - **Storage & Organization:** `Home & Kitchen -> Storage & Organization -> (Baskets, Bins & Containers, Home Storage Hooks, Racks, Shelves & Drawers, Clothing & Closet Storage)`
  - **Home Décor Products:** `Home & Kitchen -> Home Décor Products -> (Home Décor Accents, Window Treatments, Artificial Plants & Flowers, Candles & Holders, Mirrors, Rugs, Pads & Protectors, Slipcovers)`
  - **Kitchen & Dining:** `Home & Kitchen -> Kitchen & Dining -> (Dining & Entertaining, Storage & Organization, Kitchen & Table Linens)`
  - **Heating, Cooling & Air Quality:** `Home & Kitchen -> Heating, Cooling & Air Quality`
  - **Wall Art:** `Home & Kitchen -> Wall Art -> (Posters & Prints)`
  - **Bath:** `Home & Kitchen -> Bath -> (Bathroom Accessories)`
  - **Office Electronics:** `Office Products -> Office Electronics -> (Printers & Accessories)`
  - **Office & School Supplies:** `Office Products -> Office & School Supplies -> (Desk Accessories & Workspace Organizers)`
  - **Office Furniture & Lighting:** `Office Products -> Office Furniture & Lighting -> (Chairs & Sofas, Cabinets, Racks & Shelves)`
  - **Lighting & Ceiling Fans:** `Tools & Home Improvement -> Lighting & Ceiling Fans -> (Lamps & Shades, Wall Lights, Ceiling Lights)`
  - **Household Supplies:** `Health & Household -> Household Supplies -> (Household Batteries)`
  - **Health Care:** `Health & Household -> Health Care`
  - **Diet & Sports Nutrition:** `Health & Household -> Diet & Sports Nutrition -> (Nutrition Bars & Drinks)`
  - **Gardening & Lawn Care:** `Patio, Lawn & Garden -> Gardening & Lawn Care -> (Plants, Seeds & Bulbs, Pots, Planters & Container Accessories)`
  - **Patio Furniture & Accessories:** `Patio, Lawn & Garden -> Patio Furniture & Accessories`
  - **Home Audio:** `Electronics -> Home Audio -> (Home Audio Accessories, Speakers, Compact Radios & Stereos, Home Theater, Turntables & Accessories)`
  - **Video Projectors:** `Electronics -> Video Projectors`
  - **Accessories & Supplies:** `Electronics -> Accessories & Supplies -> (Audio & Video Accessories, Power Strips & Surge Protectors, Telephone Accessories)`
  - **Television & Video:** `Electronics -> Television & Video -> (Projection Screens, Televisions, DVD Players & Recorders, Streaming Media Players, Home Theater Systems, Television Accessories)`
  - **Camera & Photo:** `Electronics -> Camera & Photo -> (Tripods & Monopods, Lighting & Studio, Bags & Cases, Binoculars & Scopes, Video Surveillance, Accessories, Digital Cameras, Underwater Photography, Film Photography, Flashes, Lenses, Video)`
  - **Computers & Accessories:** `Electronics -> Computers & Accessories -> (Computer Accessories & Peripherals, Networking Products, Tablet Accessories, Computers & Tablets, Data Storage, Laptop Accessories, Computer Components)`
  - **Headphones:** `Electronics -> Headphones -> (Over-Ear Headphones, Earbud Headphones, On-Ear Headphones)`
  - **Portable Audio & Video:** `Electronics -> Portable Audio & Video -> (Boomboxes, Portable Speakers & Docks, Radios, MP3 & MP4 Player Accessories)`
  - **Security & Surveillance:** `Electronics -> Security & Surveillance -> (Accessories, Surveillance Video Equipment)`
  - **Power Accessories:** `Electronics -> Power Accessories -> (AC Adapters)`
  - **Car & Vehicle Electronics:** `Electronics -> Car & Vehicle Electronics -> (Car Electronics, Vehicle Electronics Accessories)`
  - **Wearable Technology:** `Electronics -> Wearable Technology -> (Smartwatches)`
  - **GPS, Finders & Accessories:** `Electronics -> GPS, Finders & Accessories -> (GPS System Accessories)`
  - **Accessories:** `Cell Phones & Accessories -> Accessories -> (Chargers & Power Adapters, Maintenance, Upkeep & Repairs, Single Ear Bluetooth Headsets, Smartwatch Accessories, Virtual Reality (VR) Headsets, Stands, Automobile Accessories, Photo & Video Accessories, Signal Boosters)`
  - **Cases, Holsters & Sleeves:** `Cell Phones & Accessories -> Cases, Holsters & Sleeves -> (Basic Cases, Flip Cases)`
  - **Cell Phones:** `Cell Phones & Accessories -> Cell Phones`
  - **Xbox One:** `Video Games -> Xbox One -> (Accessories)`
  - **PC:** `Video Games -> PC -> (Accessories, Virtual Reality)`
  - **Legacy Systems:** `Video Games -> Legacy Systems -> (Xbox Systems, PlayStation Systems, Nintendo Systems)`
  - **PlayStation 4:** `Video Games -> PlayStation 4 -> (Accessories)`
  - **Nintendo Switch:** `Video Games -> Nintendo Switch`
  - **Food & Beverage Gifts:** `Grocery & Gourmet Food -> Food & Beverage Gifts -> (Bakery & Dessert Gifts, Snack Gifts, Coffee & Tea Gifts, Fruit & Nut Gifts, Herb, Spice & Seasoning Gifts, Assortments & Variety Gifts, Meat & Seafood Gifts, Cheese & Charcuterie Gifts)`
  - **Breads & Bakery:** `Grocery & Gourmet Food -> Breads & Bakery -> (Cookies, Breads, Cakes, Pastries & Bakery)`
  - **Pantry Staples:** `Grocery & Gourmet Food -> Pantry Staples -> (Herbs, Spices & Seasonings, Cooking & Baking, Olives, Pickles & Relishes, Canned, Jarred & Packaged Foods, Sauces, Gravies & Marinades, Pasta & Noodles, Soups, Stocks & Broths, Dried Grains & Rice, Jams, Jellies & Sweet Spreads, Condiments & Salad Dressings, Nut & Seed Butters)`
  - **Snacks & Sweets:** `Grocery & Gourmet Food -> Snacks & Sweets -> (Snack Foods, Chocolate, Candy & Chocolate)`
  - **Dairy, Cheese & Eggs:** `Grocery & Gourmet Food -> Dairy, Cheese & Eggs -> (Cheese, Yogurt, Cheese Assortments & Samplers, Non-Dairy Milks, Milk & Cream)`
  - **Breakfast Foods:** `Grocery & Gourmet Food -> Breakfast Foods -> (Cereals, Breakfast & Cereal Bars)`
  - **Beverages:** `Grocery & Gourmet Food -> Beverages -> (Bottled Beverages, Water & Drink Mixes, Coffee, Tea & Cocoa)`
  - **Produce:** `Grocery & Gourmet Food -> Produce -> (Dried Fruits & Vegetables)`
  - **Alcoholic Beverages:** `Grocery & Gourmet Food -> Alcoholic Beverages -> (Wine)`
  - **Deli & Prepared Foods:** `Grocery & Gourmet Food -> Deli & Prepared Foods -> (Deli Meats & Cheeses)`
  - **Frozen:** `Grocery & Gourmet Food -> Frozen -> (Meals & Entrees, Meats, Ice Cream & Novelties, Appetizers & Snacks, Seafood)`
  - **Home Brewing & Winemaking:** `Grocery & Gourmet Food -> Home Brewing & Winemaking -> (Winemaking Ingredients)`
  - **Meat & Seafood:** `Grocery & Gourmet Food -> Meat & Seafood -> (Beef, Seafood)`
  - **Fresh Meal Kits:** `Grocery & Gourmet Food -> Fresh Meal Kits`
  - **Meat Substitutes:** `Grocery & Gourmet Food -> Meat Substitutes`
"""


def get_domain_specific_tips(current_url: str, host: str, original_target_url: str = "") -> str:
    """Return site-specific tips based on the current page URL.

    Args:
        current_url: The URL of the current page.
        host: The host IP/hostname to use in URL templates (e.g., "192.168.1.1").
        original_target_url: The original target URL (used for shopping tasks).
    """
    current_url = current_url.lower()
    tips_list = []

    # -------------------------------------------------------------------------
    # Shopping Admin (Adobe Commerce backend)
    # -------------------------------------------------------------------------
    if ":7780" in current_url:
        tips_list.append(f"""
# Adobe Commerce Admin Operation Expert Strategy (Your Knowledge Base and Action Guide)
Base URL: https://experienceleague.adobe.com/en/docs/commerce-admin/ (For documentation query)

**1. Core Navigation Paths (Keyword → Path):**
   - **Order Status:** `Sales -> Orders`
   - **Invoices:** `Sales -> Invoices`
   - **Reviews:** `Marketing -> All reviews`
   - **Search Terms:** `Reports -> Search Terms`
   - **Bestsellers:** `Reports -> Bestsellers`
   - **Products (Used to get and edit product attribute info):** `Catalog -> Products`
   - **Customers (Used to get and edit user info):** `Customers -> ALL Customers`
   - **Pages (Used to get page attributes like Title):** `Content -> Pages`
   - **Themes (Used to view store theme info):** `Content -> Themes`
   - **Reports Hierarchy:**
     - **Marketing:** `Reports -> Marketing`
     - **Sales:** `Reports -> Sales -> (Orders, Tax, Invoiced, Shipping, Refunds, Coupons)`
     - **Customers:** `Reports -> Customers -> (Order Total, Order Account, New, Wish Lists, Segments)`
     - **Products:** `Reports -> Products -> (Views, Bestsellers, Low Stock, Ordered, Downloads)`

**2. Common Workflows:**
   - **Check Sales:**
     1. For calculating comprehensive sales or bestsellers over two months, an efficient tip is to select `Year` for Period, rather than selecting `Month` or `Day` and calculating manually. To query sales or bestsellers for a single month, please use the `Month` selection. **This type of tip is very valuable and must be recorded.**
     2. Directly **Input (Type)** the start and end dates of the quarter in the `From` and `To` fields, instead of using the calendar selection (e.g., directly input, 01/01/22 to 03/31/22).
     3. Click `Show Report`.
     4. Scroll down to view the report.
   - **Search Product Reviews (e.g., "tanks"):**
     1. Navigate to `Marketing -> All reviews`.
     2. **Keywords**: Note that it may not be the full product name in the request. Keep keywords minimal, try searching the first word first, search for brand if available (e.g., "tank").
     3. Fill in the Product name in the Product column, do not use SKU.

**3. Critical Rules:**
   - **If valid information is not found**, try different path methods multiple times.
   - **Data Year Limitation:** The store backend only has data for **2022** and the **first five months of 2023**. Do not try to query data outside this range.
   - **Date Format:** All date input boxes must use the format `MM/DD/YY` (e.g., `03/31/22`).
   - **Completeness (Visual vs Data):** Just because a table fits in the screen doesn't mean it's complete.
     - **Mandatory Scroll Check**: After a report is displayed, you **MUST** execute at least one `scroll` action to verify integrity.
     - If the user asks for "top N" items and you see fewer than N, assume data is hidden below the fold.
     - **Action**: Unless you see "Page 1 of 1" or "End of results", strictly prioritize `scroll` to the very bottom to ensure all potential rows are rendered.
   - **Start Filling Rule:** When filling in From, **To must also be filled**. From and To can be the same, From and To include the equal case.
   - **Report Display Complete:** When you need to retrieve information in the report, you must scroll the page or content. The report may be at the bottom and make sure you scroll to the bottom of the page to see the complete report. **Unless you have clear information, all information to be retrieved must be loaded before stopping scrolling.**
   - **Report Sorting:** Report headers can be clicked to sort different options, which is very useful for requests like Newest and Oldest.
   - **Keywords:** When filling in keywords, keep the input keywords as few as possible, try to use one word. If the search fails, try using other keywords.
   - **Scroll:** Always scroll down to bottom when viewing a list/report. Do not assume data is complete without scrolling.
   - **Quantity Check**: If the user requires a specific number of items (e.g., "top 3") but you found fewer (e.g., 2), you must **strictly** verify if the page can be scrolled further.
   - **Select Option:** All UI elements with a **downward triangle symbol** are dropdown menus and should be operated using **select_option** (e.g., Order, Period etc.).
   - **Order Status:** If the user request includes a **Specified** order status (e.g., success/completed, cancelled, closed, etc.), you MUST make a selection for the order status. When you see the "Order Status" dropdown menu, you can firstly select "Specified" to see the options and then select the specific order status.

# Core Task Guidelines
*   **[Highest Priority] Apply Expert Strategy:** You **must** use the built-in **"Adobe Commerce Admin Operation Expert Strategy"** to guide your plan. Your `thought` process must explicitly state how you apply the strategy.
    *   **Step 1:** Analyze `user_query` to extract core intent.
    *   **Step 2:** Refer to **"Core Navigation Paths"** and **"Common Workflows"** to find the most matching strategy.
    *   **Step 3:** During execution, strictly abide by **"Critical Rules"** (such as date format, year limitation, completeness, etc.).
    *   **Example:** If `user_query` is "Customers are unsatisfied with tanks products, check reviews", your thinking process should be: "User intent is to query negative reviews about 'tanks'. According to expert strategy, I should execute the 'Search Product Reviews' workflow. Step 1 is to navigate to `Marketing -> All reviews`. I will generate a `click` action to click 'Marketing' in the left menu."
*   The instruction output to the executor (Grounder) must be a single atomic operation for the next step.
*   If the page does not change after entering text, try clicking nearby "Search", "Show Report" buttons.
""")

    # -------------------------------------------------------------------------
    # Shopping (One Stop Market frontend)
    # -------------------------------------------------------------------------
    if ":7770" in current_url and ":7780" not in current_url:
        tips_list.append(SHOPPING_NAVIGATION_PROMPT + f"""
**Target URL:**
{original_target_url}
If |AND| is in the target url, the first url has already been loaded in the initial page and **you need to navigate to the second url firstly.**

**Order Operation Rule:**
For order operations (e.g., buying or canceling), do NOT just add items to the cart. You MUST proceed through the entire flow: go to the shopping cart, checkout, and complete the order.

**Show products under a price:**
When asked to "show me products under a price" the benchmark evaluates based on the url. So just ensure the price is under the price asked for in the url. For example, when asked "Show me products under $25 in \\"women shoes\\" category", you go to "http://{host}:7770/clothing-shoes-jewelry/women/shoes.html?price=0-25" directly.

**For SHOPPING/BROWSING tasks:**
When asked to browse products in a particular category, navigate using the dropdown menus (not search) when possible. This may require hovering over nested dropdowns (e.g., hover over "Electronics" → hover over "Computers" → click "Laptops"). Use the hover tool to reveal these nested menus before clicking.

**For market survey tasks:**
When asked to doing a market survey, you must end with clicking to enter the product details page. For example, when asked "I am doing a market survey for one stop market, show me the most expensive product from PS4 accessories category", you must go to "http://{host}:7770/astro-gaming-a50-wireless-headset-base-station-gen-4-compatible-with-ps5-ps4-pc-mac-black-silver.html" finally.

**For viewing reviews tasks:**
You must view all the complete reviews. If there are multiple pages of comments, you must click on each page to ensure that you see all the reviews (Note: the page number is at the bottom of the page). For every page, you must scroll to the bottom of the page to see all the complete reviews.

**For Ordering Time tasks:**
When asked about the ordering time for a specific product, you must open orders from near to far according to time, and once you find the corresponding order for the target product, give the answer. For example, when asked "Tell me when I last ordered my body butter?", you should first click "my account", and then click "my order", and then click "view order" of orders with "Complete" status. If you can't find the target product in this page, you must click the next page number and repeat the above process. Once you find the corresponding order for the target product, give the answer.

**For Draft a refund message tasks:**
When asked 'Draft a refund message via their "contact us" form for the {{{{product}}}} I bought {{{{time}}}}. It broke after three days of use. The shop requires the order id, the reason and the amount to refund in the message. Don't submit yet', you must draft a message include Order Id, the reason, and the amount to refund. Specially, the reason is "It broke after three days of use.", the order id is that order number you find by the given time, and the amount is the price of the given product. For make sure of the amount to refund, you must click the "View Order" and find the true product.

**Reddit Website:**
The URL of the Reddit Website share the same ip of Target URL, but the port is 9999.
If you want to navigate to a subreddit of the Reddit Website, you directly goto the subreddit URL: http://{host}:9999/f/<subreddit_name>.
""")

    # -------------------------------------------------------------------------
    # OpenStreetMap
    # -------------------------------------------------------------------------
    if ":3000" in current_url:
        tips_list.append("""
- **For the OpenStreetMap website:**
  - **Critical Note:** On the initial OpenStreetMap page, after searching for a location, the search results will appear as links. **These links are invalid and MUST NOT be clicked!** You can only reference the link text to get detailed address information.
  - You must first extract or infer a specific location name from the user's `intent`. For example, a description like `the capital of New York State` must be replaced with `Albany` for searching or navigation. OpenStreetMap only supports searches for place names and coordinates in DD format. Vague queries like `airports that are within 50 km to CMU` are not supported and must be converted to a precise airport name, such as `Pittsburgh International Airport`, before searching. If a search fails, try modifying the query by adding or removing city, county, or neighborhood names, then search again.
  - OpenStreetMap has two search boxes: one on the initial page and another on the directions page. You must choose the appropriate search box based on the specific query.
  - For queries about distance, walking time, or driving time between two places, use the directions feature to get the answer. The entry point for the directions interface is the blue arrow button located to the right of the blue `Go` button on the initial page.
  - For queries like `Which US states border Vermont?`, since the map data may be incomplete, simply search for `Vermont` in the search box and then answer directly. Your answer should only consider states with a direct land border.
  - To select the mode of transport on the directions page, use the select_option action for the dropdown menu that appears after clicking. You must strictly use the options provided by the website and not guess. The only valid transport options are Car (OSRM), Bicycle (OSRM), and Foot (OSRM).
  - For queries asking if it's possible to travel from `place1` to `place2` within a specified time, you must use the navigation feature to determine the answer.
  - For queries about the walking or driving time from place1 to place2, you must use the directions feature to obtain the answer. The answer should be formatted as X hours Y minutes, e.g., 1 hour 25 minutes.
  - To zoom in or out on the map, **do not use `scroll up` or `scroll down`**. You must use the `zoom_in` and `zoom_out` actions, passing the desired zoom level in the `level` parameter.
  - For queries asking for a `zip code`, search for the location on the initial page. The zip code can be found in the search result string. For example, if the search result for CMU is `Carnegie Mellon University, ..., 15232, United States`, you should directly output `15232`.
  - If you need the coordinates of a location, first search for it on the initial page. Then, find the target entry in the search results and hover your mouse over it. A location marker will appear on the map. Right-click on this marker and select `Show Address` to get the coordinates in Decimal Degrees (DD) format.
  - For queries about a detailed address (e.g., the address of an international airport), you must search for the location on the initial OpenStreetMap page. **Prioritize using the address from the resulting link text.** Even if you can visually locate the target on the map, you MUST still perform a search to use the official address from the search result. If the location cannot be found via search, use the following format as a fallback: `Institution/Place Name, Street Name, Neighborhood/District Name, City, County, State, ZIP Code, Country`. For example: `Carnegie Mellon University, Canterbury Lane, Shadyside, Pittsburgh, Allegheny County, Pennsylvania, 15232, United States`.
  - For queries like `the closest/nearest [place1] to [place2]`, combine your prior knowledge with map data. If you already know the specific name of the closest place, use that name directly. If you do not know, you must first search for `[place2]` and then visually scan the map to find the closest `[place1]`.
  - For queries like Pull up the description page of {{{{location}}}} on Map, the task is complete once you have searched for the location on the initial OpenStreetMap page.
  - On the directions page, if the Go button has no effect after you've entered the start and end points, it means one of the locations is invalid. You should correct the name and try again. To verify a location's name, you can search for it on the initial OpenStreetMap page first, and then use the confirmed name on the directions page.
  - When searching for locations, do not count coffee shops as restaurants.
""")

    # -------------------------------------------------------------------------
    # Reddit (Postmill)
    # -------------------------------------------------------------------------
    if ":9999" in current_url:
        tips_list.append(f"""
- **For the Reddit website:**
    - To access a user's profile page, the link is http://{host}:9999/user/{{{{username}}}}
    - To view your own profile page (forum), the link is http://{host}:9999/user/MarvelsGrantMan136
    - To view all forums, the link is http://{host}:9999/forums/all
    - To view a specific forum, the link is http://{host}:9999/f/{{{{forum_name}}}} (however, if it results in a 404 error URL, please go to http://{host}:9999/forums/all instead)
    - To view the comments on a post (submission), click directly on "xxx comments" or "no comments" below the post title. Do not click the title itself, as it may redirect you to the image link or another web link contained in the post. Conversely, if you want to view the post's image, click the title.
    - To edit your own post, click "Edit" below the title, not the title itself.
    - Reposting is not currently supported. You must submit a new post to do this. If the original post contains an image, you need to first open the image, record its link, and then paste it into the URL field when submitting the new post.
    - If a task mentions "top n posts," it refers to the top n posts based on the default sorting order (i.e., hot).
    - If a task mentions something like "in r/sports," it refers to the "sports" forum. The same logic applies to other forums.
    - The term "subreddit" is used to mean "forum"
    - To access GitLab, the link is http://{host}:8023
    - To access a specific GitLab repository {{{{username}}}}/{{{{reponame}}}}, the link is http://{host}:8023/{{{{username}}}}/{{{{reponame}}}}
""")

    # -------------------------------------------------------------------------
    # GitLab
    # -------------------------------------------------------------------------
    if ":8023" in current_url:
        tips_list.append(f"""
- **For the GitLab website:**
    - URL-First Strategy is Key: For GitLab, your primary strategy should always be to directly construct a URL and use the goto action, as this is far more efficient than UI navigation. Only if a constructed URL fails should you resort to step-by-step GUI operations. When constructing URLs, ensure you use the correct host address (http://{host}:8023).
    - Constructing Issue URLs:
        - All issue-related tasks can be solved with a direct URL. The structure is http://{host}:8023/<username>/<reponame>/-/issues/?<parameters>.
        - Parameter Order: You must follow a strict parameter order: sort first, then state, then label_name.
        - Translate Intent to Terms: You must translate natural language from the intent into GitLab's specific terminology. For example, help needed must be converted to help wanted, and bug must be capitalized to BUG.
        - Specific Filters:
            - To Check out the most recent open issues, go to the URL .../issues/?sort=created_date&state=opened.
            - To List all opened issues that don't have any labels, use the filter .../issues/?label_name%5B%5D=None.
            - For intents like Display the list of issues... with labels related to questions, directly goto .../issues/?label_name%5B%5D=question.
    - Interacting with Merge Requests:
        - For tasks like Post "{{{{content}}}}" for the merge request..., after filling in the content, you must click the 'Comment' button. Do not click the 'Comment & close merge request' button unless explicitly instructed to do so.
        - For tasks like Go to the merge request on {{{{topic}}}} I have to review..., you must start by clicking the 'Merge requests' icon in the top-right corner of the dashboard. You must check both the 'Assigned to you' and 'Review requests for you' sections to find the correct MR. To decide on your reply, check the activity feed: if it only contains a system message (e.g., user2 assigned...), reply with a simple @ mention; if there is other text content from the author, reply with Thank you.
    - Major UI Workflows:
        - Forking All Repositories: For tasks requiring you to fork all of a user's repositories, you must follow a multi-step process: first, navigate to the user's profile page, then locate their list of personal projects, and finally, iterate through the list, forking each project one by one.
        - Searching for Repositories: The search function does not support the user/reponame format. To find a repository like kkroening/ffmpeg-python, you must search for ffmpeg-python only and then locate the project owned by kkroening in the search results.
        - Finding and Following Users: For tasks like Follow {{{{account_list}}}}..., if a directly constructed profile URL results in a 404 error, you must use the search bar on the dashboard. After searching, click on the 'Users' filter in the sidebar to locate the correct user profile.
        - Handling User/Repo Names (URL Construction Priority): The displayed username (e.g., Byte Blaze) might differ from the actual name used in the URL path. If unsure, first navigate by clicking to the repository page. Then, get the correct username from the page's URL. Your next priority is to use this correct username to construct a new goto URL. Only if that new URL fails should you fall back to further GUI operations.
    - Cloning Repositories:
        - To find the clone URL, first check the project's README. If not there, click the "Clone" button. Use hscroll to scroll horizontally if the full URL is not visible.
        - For questions like Show me the command to clone {{{{repo}}}} with SSH, you must find the SSH clone URL and then replace the IP address with metis.lti.cs.cmu.edu in the final command. For example, git clone ssh://git@{host}:2222/... must be changed to git clone ssh://git@metis.lti.cs.cmu.edu:2222/....
    - Finding Your Assigned Items:
        - To find issues, merge requests, or to-do items assigned to you, use the icons in the top-right corner of the dashboard.
        - For complex queries like Open my latest created issue that has <keyword> in its title, you can either click the "Issues" icon and search, or directly goto a URL like http://{host}:8023/dashboard/issues?scope=all&state=opened&assignee_username=<your_username>&search=<keyword>.
    - Specific Answer Formatting:
        - For questions about people (who has made the most contributions, who else has access), your final answer must only be the person's name(s).
        - For tasks like ...to check if it is closed, you must first navigate to the specific issue's page (e.g., .../-/issues/719), not the search results page. Your final answer must then be only "Yes" or "No".
        - For tasks like Create a repo named ... with movies directed by ... in a README file, you must list all qualifying movies in the README. The list should contain only the movie titles, with no other information like release years.
    - Misc UI Tips:
        - To find public repositories, use the Explore section on the main page.
        - To get a precise timestamp from relative times (e.g., "updated 2 years ago"), hover your mouse over the text.
        - When setting your status to 'Busy', do not check the 'Set yourself as busy' checkbox. Instead, you must type Busy directly into the status input field.
- **For Cross-Website Reddit Queries:**
        - For queries related to subreddits, such as "...URLs of the 5 most recent posts from the movies?" or "the most active {{{{num}}}} DIY ideas on DIY subreddit?", you must find the answers on the internally deployed Reddit website at http://{host}:9999. For each qualifying result, the required output is the URL of its comments page, for example: http://{host}:9999/f/news/129905/ohio-man-charged-for-using-molotov-cocktails-to-attack.
        - For the Reddit website:
            - To access a user's profile page, the link is http://{host}:9999/user/username
            - To view your own profile page, the link is http://{host}:9999/user/MarvelsGrantMan136
            - If a user_query mentions "top n posts," it refers to the top n posts based on the default sorting order (i.e., hot).
            - If a user_query mentions something like "in r/sports," it refers to the "sports" forum. The same logic applies to other forums.
            - The term "subreddit" is used to mean "forum"
            - To view all forums, the link is http://{host}:9999/forums/all, if the user_query doesn't specify a forum, please go to this url to find the most relevant forum.
            - To view a specific forum, the link is http://{host}:9999/f/forum_name (however, if it results in a 404 error URL, please go to http://{host}:9999/forums/all instead)
            - To view the comments on a post (submission), click directly on "xxx comments" or "no comments" below the post title. Do not click the title itself, as it may redirect you to the image link or another web link contained in the post. Conversely, if you want to view the post's image, click the title.
            - To edit your own post, click "Edit" below the title, not the title itself.
            - Reposting is not currently supported. You must submit a new post to do this. If the original post contains an image, you need to first open the image, record its link, and then paste it into the URL field when submitting the new post.
""")

    if not tips_list:
        return "None"

    return "\n".join(tips_list)


# =============================================================================
# Site-Specific Summary Tips (for Summary module)
# =============================================================================

def get_summary_tips(current_url: str, host: str) -> str:
    """Return site-specific tips for the Summary module.

    Args:
        current_url: The URL of the current page.
        host: The host IP/hostname to use in URL templates.
    """
    current_url = current_url.lower()
    tips_list = []

    # OpenStreetMap
    if ":3000" in current_url:
        tips_list.append("""
- **For OpenStreetMap tasks:**
    - For general knowledge queries like `Which US states border Vermont`, answer directly based on your geographical knowledge, using the map in the screenshot for confirmation. Your answer should only consider states with a direct land border.
    - For questions about a detailed address, prioritize using the address found in the text of the OpenStreetMap search result link (which should be visible in the final screenshots). If no search result is found in the history, use your prior knowledge to construct the address in the following format as a fallback: Institution/Place Name, Street Name, Neighborhood/District Name, City, County, State, ZIP Code, Country. For example: `Carnegie Mellon University, Canterbury Lane, Shadyside, Pittsburgh, Allegheny County, Pennsylvania, 15232, United States`.
    - For zip code queries, prioritize extracting the code directly from the OpenStreetMap search result visible in the screenshot. For example, extract `15232` from `Carnegie Mellon University, ..., 15232, United States`.
    - For queries about coordinates in Decimal Degrees (DD) format, prioritize extracting them from the URL of the search result or from information displayed directly on the page (visible in the screenshots). If the coordinates cannot be found in the provided context, use your prior knowledge to provide the answer. The final coordinates must be rounded to three decimal places. For example, if OpenStreetMap provides `40.46081, -79.94668`, your output must be `40.460, -79.946`.
    - When searching for locations, do not count coffee shops as restaurants.
    - The travel time displayed in OpenStreetMap's directions is in an HH:MM format (e.g., 01:25, meaning 1 hour and 25 minutes). You must convert this into the X h Y min format for your final answer (e.g., 1 h 25 min).
    - For queries like Measure distance between {{{{location/address_1}}}} and {{{{location/address_2}}}} by walking, you should only return the distance from the navigation results, for example, 1.2km.
    - Strict Format Matching: If the example shows "557m", please use the exact same format. If it is name information like product user, return only the name in the answer, do not return other information.
""")

    # GitLab
    if ":8023" in current_url:
        tips_list.append(f"""
- **For the GitLab website:**
    - Specific Answer Formatting:
        - Output Formatting for Clone Commands: For questions like Show me the command to clone <repo> with SSH, you must find the SSH clone URL and then replace the IP address with `metis.lti.cs.cmu.edu` in the final command. For example, `git clone ssh://git@{host}:2222/yjlou/2019-nCov.git` must be changed to `git clone ssh://git@metis.lti.cs.cmu.edu:2222/yjlou/2019-nCov.git`.
        - For questions about people, like `who has made the most contributions` or `who else has access`, your final answer must only be the person's name(s). Do not include any other text.
        - For tasks like `...to check if it is closed`, your final answer must be only "Yes" or "No". Do not include any other text.
        - For queries related to subreddits, such as "...URLs of the 5 most recent posts from the movies?" or "the most active {{{{num}}}} DIY ideas on DIY subreddit?", you must find the answers on the internally deployed Reddit website at http://{host}:9999. For each qualifying result, the required output is the URL of its comments page, for example: http://{host}:9999/f/news/129905/ohio-man-charged-for-using-molotov-cocktails-to-attack.
""")

    # Shopping / Shopping Admin (shared answer format)
    if ":7770" in current_url or ":7780" in current_url:
        tips_list.append("""
════════════════════════════════════════════════════════════════════════
Answer Format (Very Important):
════════════════════════════════════════════════════════════════════════

When you generate the final answer, please follow these principles:

1. **Strict Format Matching**: If the example shows "557m", please use the exact same format. If it is name information like product user, return only the name in the answer, do not return other information.
2. **Provide Complete Answer**: Include enough context so that the answer can stand alone.
3. **Add Reasoning Appropriately**: For questions requiring judgment (yes/no, status check, comparison), include short context or reasoning next to the answer.
4. **Accurate Terminology**: When copying text, use the exact wording from the source file.
5. **Ties**: When involving sorting or Top N questions, if there are numerical ties, you must list **all tied candidates** in the answer, even if this causes the final result quantity to exceed N. Please be sure to **carefully check the values**, do not just intercept the first N rows of the report, because the sorting of items with the same value in the report may be random, you must check if there are more items with the same value ranked behind.
6. When asked to return the answer in MM:COUNT format, please return it like this: "January: 1". It expects MM to be the explicit name of the month rather than a number.
7. When asked how much it costs, return only the decimal. Therefore, if the item costs $7.50, return "7.50"; if it costs $0, return "0".
8. When asked about configuration, return 2x2 instead of 2*2.
9. If there are multiple matching entries for amount-based questions, please list each amount in the reasoning and ensure that the final answer string contains the aggregated total satisfying the query (e.g., sum of all matching refunds).
""")

    if not tips_list:
        return "None"

    return "\n".join(tips_list)