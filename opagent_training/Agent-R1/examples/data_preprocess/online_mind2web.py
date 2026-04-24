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

def read_json_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        outputs = json.load(f)
    return outputs

def get_tasks_with_correct_answer(input_file, model_type='Operator'):
    human_labels = read_json_file(input_file)
    tasks_list = []
    for task_label_i in human_labels:
        if task_label_i[f"{model_type}_human_label"] == "1":
            tasks_list.append(task_label_i["task_id"])
    return tasks_list

def list2dict_bykey(input_list, model_type):
    output_dict = {}
    for input_i in input_list:
        output_dict[input_i[model_type]] = input_i
    return output_dict

def collect_correct_answers(dataset_path, resuls_path, model_type='Operator'):
    task_list = read_json_file(os.path.join(dataset_path, "Online_Mind2Web.json"))
    task_dict = list2dict_bykey(task_list, "task_id")
    model_results = read_json_file(os.path.join(resuls_path, "evaluation_results", f"{model_type.lower()}_results.json"))
    results_dict = list2dict_bykey(model_results, "task_id")
    tasks_list_with_correct_answer = get_tasks_with_correct_answer(os.path.join(resuls_path, "human_labels.json"), model_type)
    output_task_list = []
    for task_i_corr_ans in tasks_list_with_correct_answer:
        task_i = task_dict[task_i_corr_ans]
        results_dict_i = results_dict[task_i_corr_ans]




if __name__ == '__main__':
    online_mind2web_results_path = "<ONLINE_MIND2WEB_RESULTS_DIR>"
    local_dir = "<ONLINE_MIND2WEB_LOCAL_DIR>"
    data_source = 'osunlp/Multimodal-Mind2Web'

    
        
    # Inspect the structure of the first item to understand the data format
    print("Sample data structure:", json.dumps(train_data[0], indent=2)[:500] + "...")
    
    # Convert to datasets format with proper type handling
    def process_supporting_facts(facts):
        # Convert supporting facts to a serializable format
        return json.dumps(facts)
    
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
