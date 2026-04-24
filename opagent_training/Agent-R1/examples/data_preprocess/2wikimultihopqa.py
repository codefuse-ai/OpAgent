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
Preprocess the 2WikiMultihopQA dataset to parquet format
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
    parser.add_argument('--local_dir', default='~/data/2wikimultihopqa')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--download_method', choices=['huggingface', 'direct'], default='huggingface',
                        help='Method to download the dataset: huggingface or direct')
    parser.add_argument('--train_size', type=int, default=12800,
                        help='Number of training samples to use')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of validation samples to use')

    args = parser.parse_args()

    data_source = 'xanhho/2WikiMultihopQA'
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Load the 2WikiMultihopQA dataset
    if args.download_method == 'huggingface':
        try:
            print("Loading 2WikiMultihopQA dataset from Hugging Face...")
            dataset = datasets.load_dataset(data_source, trust_remote_code=True)
            train_dataset = dataset['train']
            validation_dataset = dataset['validation']
            test_dataset = dataset.get('test', None)
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            print("Please try using the direct download method with --download_method=direct")
            exit(1)
    else:
        # Direct download from the 2WikiMultihopQA source
        print("Downloading 2WikiMultihopQA dataset directly...")
        
        # URLs for the dataset files - these would need to be updated with actual URLs
        train_url = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/train.parquet?download=true"
        dev_url = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet?download=true"
        test_url = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/test.parquet?download=true"
        
        train_file = os.path.join(local_dir, "train.parquet")
        dev_file = os.path.join(local_dir, "dev.parquet")
        test_file = os.path.join(local_dir, "test.parquet")
        
        # Download files if they don't exist
        if not os.path.exists(train_file):
            print(f"Downloading training data to {train_file}...")
            download_file(train_url, train_file)
        
        if not os.path.exists(dev_file):
            print(f"Downloading validation data to {dev_file}...")
            download_file(dev_url, dev_file)
        
        if not os.path.exists(test_file):
            print(f"Downloading test data to {test_file}...")
            download_file(test_url, test_file)
        
        # Load the downloaded files
        print("Loading downloaded files...")
        train_dataset = datasets.Dataset.from_parquet(train_file)
        validation_dataset = datasets.Dataset.from_parquet(dev_file)
        test_dataset = datasets.Dataset.from_parquet(test_file) if os.path.exists(test_file) else None

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
            
            # Process supporting facts if available
            supporting_facts = example.get('supporting_facts', [])
            if isinstance(supporting_facts, str):
                try:
                    supporting_facts = json.loads(supporting_facts)
                except (json.JSONDecodeError, TypeError):
                    supporting_facts = []
            
            # Extract evidence information if available
            evidences = example.get('evidences', [])
            if isinstance(evidences, str):
                try:
                    evidences = json.loads(evidences)
                except (json.JSONDecodeError, TypeError):
                    evidences = []
            
            # Extract entity IDs if available
            entity_ids = example.get('entity_ids', '')
            
            # Extract type information
            question_type = example.get('type', '')
            
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
                    'index': str(idx),
                    'answer': answer_raw,
                    'question': question_raw,
                    'supporting_facts': json.dumps(supporting_facts),
                    'evidences': json.dumps(evidences),
                    'entity_ids': entity_ids,
                    'type': question_type
                }
            }
            return data

        return process_fn

    # Map the datasets
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)
    
    # Save to parquet
    train_dataset.to_parquet(os.path.join(local_dir, 'train_processed.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation_processed.parquet'))
    
    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        test_dataset.to_parquet(os.path.join(local_dir, 'test_processed.parquet'))

    # Copy to HDFS if needed
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir) 