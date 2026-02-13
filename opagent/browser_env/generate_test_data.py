"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json
import os

from env_config import *


def main() -> None:
    DATASET = os.environ["DATASET"]
    # TODO: 配置您的输入输出路径
    input_path = os.getenv("INPUT_PATH", "./config_files/wa/")
    output_path = os.getenv("OUTPUT_PATH", "./config_files/wa_ali/")
    if DATASET == "webarena":
        print("DATASET: webarena")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"SHOPPING_ADMIN: {SHOPPING_ADMIN}")
        print(f"GITLAB: {GITLAB}")
        print(f"WIKIPEDIA: {WIKIPEDIA}")
        print(f"MAP: {MAP}")
        print(f"HOMEPAGE: {HOMEPAGE}")
        inp_paths = [f"{input_path}/test_webarena.raw.json"]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
            "__GITLAB__": GITLAB,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__MAP__": MAP,
            "__HOMEPAGE__": HOMEPAGE,
        }
    elif DATASET == "visualwebarena":
        print("DATASET: visualwebarena")
        print(f"CLASSIFIEDS: {CLASSIFIEDS}")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"HOMEPAGE: {HOMEPAGE}")
        inp_paths = [
            f"{input_path}/test_classifieds.raw.json", f"{input_path}/test_shopping.raw.json", f"{input_path}/test_reddit.raw.json",
        ]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__CLASSIFIEDS__": CLASSIFIEDS,
            "__HOMEPAGE__": HOMEPAGE,
        }
    else:
        raise ValueError(f"Dataset not implemented: {DATASET}")
        
    for inp_path in inp_paths:
        output_dir = inp_path.replace('.raw.json', '')
        output_dir = output_dir.replace('/wa/', '/wa_ali/')
        #output_dir = f"{output_path}/{output_dir}"
        os.makedirs(output_dir, exist_ok=True)
        with open(inp_path, "r") as f:
            raw = f.read()
        for k, v in replace_map.items():
            raw = raw.replace(k, v)

        with open(inp_path.replace(".raw", ""), "w") as f:
            f.write(raw)
        data = json.loads(raw)
        for idx, item in enumerate(data):
            with open(os.path.join(output_dir, f"{idx}.json"), "w") as f:
                json.dump(item, f, indent=2)


if __name__ == "__main__":
    main()