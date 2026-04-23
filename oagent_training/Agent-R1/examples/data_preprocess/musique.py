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
Preprocess the MusiQue dataset to parquet format
"""

import os
import datasets
import argparse
import json
import requests
from tqdm import tqdm
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
    parser.add_argument('--local_dir', default='~/data/musique')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--download_method', choices=['huggingface', 'direct'], default='huggingface',
                        help='Method to download the dataset: huggingface or direct')
    parser.add_argument('--config', choices=['default', 'answerable'], default='default',
                        help='Dataset configuration to use: default (MusiQue-Full) or answerable (MusiQue-Ans)')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Number of training samples to use')
    parser.add_argument('--val_size', type=int, default=None,
                        help='Number of validation samples to use')

    args = parser.parse_args()

    data_source = 'bdsaglam/musique'
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Load the MusiQue dataset
    if args.download_method == 'huggingface':
        try:
            print(f"Loading MusiQue dataset from Hugging Face (config: {args.config})...")
            dataset = datasets.load_dataset(data_source, args.config)
            train_dataset = dataset['train']
            validation_dataset = dataset['validation']
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            print("Please try using the direct download method with --download_method=direct")
            exit(1)
    else:
        # Direct download from the MusiQue source
        print(f"Downloading MusiQue dataset directly (config: {args.config})...")
        
        # URLs for the dataset files - these would need to be updated with actual URLs
        if args.config == 'default':
            train_url = "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_full_v1.0_train.jsonl?download=true"
            dev_url = "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_full_v1.0_dev.jsonl?download=true"
        else:  # answerable
            train_url = "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_ans_v1.0_train.jsonl?download=true"
            dev_url = "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_ans_v1.0_dev.jsonl?download=true"
        
        train_file = os.path.join(local_dir, f"musique_{args.config}_train.jsonl")
        dev_file = os.path.join(local_dir, f"musique_{args.config}_dev.jsonl")
        
        # Download files if they don't exist
        if not os.path.exists(train_file):
            print(f"Downloading training data to {train_file}...")
            download_file(train_url, train_file)
        
        if not os.path.exists(dev_file):
            print(f"Downloading validation data to {dev_file}...")
            download_file(dev_url, dev_file)
        
        # Load the downloaded files
        print("Loading downloaded files...")
        train_dataset = datasets.Dataset.from_json(train_file)
        validation_dataset = datasets.Dataset.from_json(dev_file)

    # Sample the datasets if needed
    if args.train_size is not None and args.train_size < len(train_dataset):
        indices = random.sample(range(len(train_dataset)), args.train_size)
        train_dataset = train_dataset.select(indices)
    if args.val_size is not None and args.val_size < len(validation_dataset):
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
            # Extract the question
            question_raw = example.get('question', '')
            question = instruction_following + "Question: " + question_raw
            
            # Extract the answer
            answer_raw = example.get('answer', '')
            
            # Extract supporting facts if available
            supporting_facts = example.get('supporting_facts', [])
            if isinstance(supporting_facts, str):
                try:
                    supporting_facts = json.loads(supporting_facts)
                except (json.JSONDecodeError, TypeError):
                    supporting_facts = []
            
            # Extract context if available
            context = example.get('context', [])
            if isinstance(context, str):
                try:
                    context = json.loads(context)
                except (json.JSONDecodeError, TypeError):
                    context = []
            
            # Extract decomposition if available
            decomposition = example.get('decomposition', [])
            if isinstance(decomposition, str):
                try:
                    decomposition = json.loads(decomposition)
                except (json.JSONDecodeError, TypeError):
                    decomposition = []
            
            # Extract bridge entities if available
            bridge_entities = example.get('bridge_entities', [])
            if isinstance(bridge_entities, str):
                try:
                    bridge_entities = json.loads(bridge_entities)
                except (json.JSONDecodeError, TypeError):
                    bridge_entities = []
            
            # Extract id
            example_id = example.get('id', str(idx))
            
            # Extract answerable flag
            answerable = example.get('answerable', True)
            
            # Create the data structure
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
                    'id': example_id,
                    'index': str(idx),
                    'answer': answer_raw,
                    'question': question_raw,
                    'supporting_facts': json.dumps(supporting_facts),
                    'context': json.dumps(context),
                    'decomposition': json.dumps(decomposition),
                    'bridge_entities': json.dumps(bridge_entities),
                    'answerable': answerable,
                    'config': args.config
                }
            }
            return data

        return process_fn

    # Map the datasets
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)
    
    # Save to parquet
    train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.config}_processed.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, f'validation_{args.config}_processed.parquet'))
    
    # Copy to HDFS if needed
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir) 