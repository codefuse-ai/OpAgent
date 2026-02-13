import os
import re

TARGET_DIR = "/ossfs/workspace/oagent_training_onlineRL/Agent-R1/eval/demo"

replacements = [
    # browser_env related
    (r"from agent_r1\.tool\.tools\.browser_env", "from demo.browser_env"),
    (r"import agent_r1\.tool\.tools\.browser_env", "import demo.browser_env"),
    (r"from \.browser_env", "from demo.browser_env"), # In case of relative import issues from parent
    
    # evaluation_harness related
    (r"from agent_r1\.tool\.tools\.evaluation_harness", "from demo.evaluation_harness"),
    (r"import agent_r1\.tool\.tools\.evaluation_harness", "import demo.evaluation_harness"),
    (r"from demo.evaluation_harness", "from demo.evaluation_harness"),

    # tool utils
    (r"from agent_r1\.tool\.utils", "from demo.tool_utils"),
    (r"import agent_r1\.tool\.utils", "import demo.tool_utils"),
    
    # Fix internal package references if they used absolute paths
    (r"from agent_r1\.tool\.tools\.browser_env\.actions", "from demo.browser_env.actions"),
    (r"from agent_r1\.tool\.tools\.browser_env\.utils", "from demo.browser_env.utils"),
    (r"from agent_r1\.tool\.tools\.browser_env\.constants", "from demo.browser_env.constants"),
    
    # webjudge related
    (r"from agent_r1\.tool\.tools\.evaluation_harness\.webjudge", "from demo.evaluation_harness.webjudge"),
]

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Special handling for relative imports within the packages if they are broken
    # For now, let's trust that intra-package relative imports (from . import x) work fine.
    
    if content != original_content:
        print(f"Updated {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".py"):
                process_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
