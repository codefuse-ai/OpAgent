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
Preprocess the OnlineMind2Web dataset to parquet format
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

def get_instruction_prompt():
    instruction_following = """You are a Web Agent focused on operating the web browser. Given the user query, your goal is to generate accurate web operation instructions to complete and answer the user query.
To fulfill the user's query, you may need to perform a series of single-step operations on the website to reach the target page and answer the user's question.
You must first conduct reasoning inside <think>...</think>. If you need to operate the web browser, you can generate a single-step operation information inside <web_operation>...</web_operation> to control the web browser after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<web_operation>
...
</web_operation>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""    
    return instruction_following

def read_json_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        outputs = json.load(f)
    return outputs

def process_webarena_gt(input_file):
    webarena_dataset = []
    test_file_list = os.listdir(input_file)
    for test_file in test_file_list:
        test_sample_ann_i = read_json_file(os.path.join(input_file, test_file))
        webarena_dataset.append(test_sample_ann_i)


def save_dataset2parquet(train_data, validation_data, output_path):
    train_dataset = datasets.Dataset.from_dict({
        'question': [item['question'] for item in train_data],
        'answer': [item['answer'] for item in train_data],
        'level': [str(item.get('level', '')) for item in train_data],
        'type': [str(item.get('type', '')) for item in train_data]
    })
    
    validation_dataset = datasets.Dataset.from_dict({
        'question': [item['question'] for item in validation_data],
        'answer': [item['answer'] for item in validation_data],
        'level': [str(item.get('level', '')) for item in validation_data],
        'type': [str(item.get('type', '')) for item in validation_data]
    })
    
                         

    # Process each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question')
            question = get_instruction_prompt() + "User Query: " + question_raw
            
            answer_raw = example.pop('answer')
            
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
                    'level': str(example.get('level', '')),
                    'type': str(example.get('type', ''))
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)
    
    train_dataset.to_parquet(os.path.join(output_path, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(output_path, 'validation.parquet'))


if __name__ == '__main__':
    visual_webarena_dataset_path = "<VISUAL_WEBARENA_DATASET_DIR>"
    output_path = "<OUTPUT_DATA_DIR>"
    data_source = 'osunlp/Multimodal-Mind2Web'


    
    

