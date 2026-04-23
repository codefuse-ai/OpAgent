# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the HotpotQA dataset to parquet format
"""

import os
import datasets
import argparse
import json
import requests
from tqdm import tqdm
import zipfile
import random
from verl.utils.hdfs_io import copy, makedirs


def download_file(url, local_path):
    """Download a file from a URL to a local path with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(local_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/hotpotqa')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--download_method', choices=['huggingface', 'direct'], default='direct',
                        help='Method to download the dataset: huggingface or direct')
    parser.add_argument('--train_size', type=int, default=12800,
                        help='Number of training samples to use')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of validation samples to use')

    args = parser.parse_args()

    data_source = 'hotpotqa/hotpot_qa'
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Load the HotpotQA dataset
    if args.download_method == 'huggingface':
        try:
            dataset = datasets.load_dataset(data_source, trust_remote_code=True)
            train_dataset = dataset['train']
            validation_dataset = dataset['validation']
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            print("Please try using the direct download method with --download_method=direct")
            exit(1)
    else:
        # Direct download from the HotpotQA source
        print("Downloading HotpotQA dataset directly from source...")
        
        # URLs for the dataset files
        train_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
        dev_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        
        train_file = os.path.join(local_dir, "hotpot_train_v1.json")
        dev_file = os.path.join(local_dir, "hotpot_dev_distractor_v1.json")
        
        # Download files if they don't exist
        if not os.path.exists(train_file):
            print(f"Downloading training data to {train_file}...")
            download_file(train_url, train_file)
        
        if not os.path.exists(dev_file):
            print(f"Downloading validation data to {dev_file}...")
            download_file(dev_url, dev_file)
        
        # Load the downloaded files
        print("Loading downloaded files...")
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(dev_file, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)
        
        # Inspect the structure of the first item to understand the data format
        print("Sample data structure:", json.dumps(train_data[0], indent=2)[:500] + "...")
        
        # Convert to datasets format with proper type handling
        def process_supporting_facts(facts):
            # Convert supporting facts to a serializable format
            return json.dumps(facts)
        
        train_dataset = datasets.Dataset.from_dict({
            'question': [item['question'] for item in train_data],
            'answer': [item['answer'] for item in train_data],
            'supporting_facts': [process_supporting_facts(item.get('supporting_facts', [])) for item in train_data],
            'level': [str(item.get('level', '')) for item in train_data],
            'type': [str(item.get('type', '')) for item in train_data]
        })
        
        validation_dataset = datasets.Dataset.from_dict({
            'question': [item['question'] for item in validation_data],
            'answer': [item['answer'] for item in validation_data],
            'supporting_facts': [process_supporting_facts(item.get('supporting_facts', [])) for item in validation_data],
            'level': [str(item.get('level', '')) for item in validation_data],
            'type': [str(item.get('type', '')) for item in validation_data]
        })

        if args.train_size is not None:
            indices = random.sample(range(len(train_dataset)), args.train_size)
            train_dataset = train_dataset.select(indices)
        if args.val_size is not None:
            indices = random.sample(range(len(validation_dataset)), args.val_size)
            validation_dataset = validation_dataset.select(indices)
    
    instruction_following = """Answer the given question. You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""                             

    # Process each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question')
            question = instruction_following + "Question: " + question_raw
            
            answer_raw = example.pop('answer')
            
            # Parse the supporting facts from JSON string back to Python object if needed
            supporting_facts_str = example.get('supporting_facts', '[]')
            try:
                supporting_facts = json.loads(supporting_facts_str)
            except (json.JSONDecodeError, TypeError):
                supporting_facts = []
            
            # Convert all data to string format to avoid type issues
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "multihop_qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'split': split,
                    'index': str(idx),
                    'answer': answer_raw,
                    'question': question_raw,
                    'supporting_facts': json.dumps(supporting_facts),  # Store as JSON string
                    'level': str(example.get('level', '')),
                    'type': str(example.get('type', ''))
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
