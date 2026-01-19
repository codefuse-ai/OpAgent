import json
import os
from collections import defaultdict
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self.conflict_groups = defaultdict(list)
        self.non_conflicting = []
    
    def load_tasks(self, input_path: str, input_list: List[str] = None) -> None:
        """加载任务"""
        task_file_names = input_list
        for task_file_name in task_file_names:
            full_path = os.path.join(input_path, task_file_name)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    task_conf = json.load(f)
                
                conflict_key = task_conf.get('conflict_key')
                
                if conflict_key and conflict_key != "":
                    self.conflict_groups[conflict_key].append({
                        'file': task_file_name,
                        'config': task_conf,
                        'conflict_key': conflict_key
                    })
                else:
                    self.non_conflicting.append({
                        'file': task_file_name,
                        'config': task_conf,
                        'conflict_key': None
                    })
            except Exception as e:
                logger.error(f"Error loading {task_file_name}: {e}")
    
    def schedule_tasks_fixed_batches_balanced(self, num_batches: int = 7) -> List[List[Dict]]:
        total_tasks = sum(len(tasks) for tasks in self.conflict_groups.values()) + \
                      len(self.non_conflicting)
        batches = [[] for _ in range(num_batches)]
        conflict_items = sorted(
            self.conflict_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        max_tasks_per_key = max((len(t) for _, t in conflict_items), default=0)
        task_queue = []
        for position in range(max_tasks_per_key):
            for conflict_key, tasks in conflict_items:
                if position < len(tasks):
                    task_queue.append(tasks[position])
        
        for task in task_queue:
            available_batches = [i for i in range(num_batches) 
                                if len(batches[i]) < self.batch_size]
            if available_batches:
                best_batch_idx = min(available_batches, key=lambda i: len(batches[i]))
            else:
                best_batch_idx = min(range(num_batches), key=lambda i: len(batches[i]))
            batches[best_batch_idx].append(task)
        
        for task in self.non_conflicting:
            available_batches = [i for i in range(num_batches) 
                                if len(batches[i]) < self.batch_size]
            if available_batches:
                best_batch_idx = min(available_batches, key=lambda i: len(batches[i]))
            else:
                best_batch_idx = min(range(num_batches), key=lambda i: len(batches[i]))
            batches[best_batch_idx].append(task)
        return batches
    
    def analyze_ecs_required(self, batches: List[List[Dict]]) -> Dict:
        max_ecs_needed = 0
        batch_ecs_list = []
        for batch_idx, batch in enumerate(batches, 1):
            conflict_count = defaultdict(int)
            for task in batch:
                if task['conflict_key']:
                    conflict_count[task['conflict_key']] += 1
            if conflict_count:
                max_repeat = max(conflict_count.values())
            else:
                max_repeat = 0
            batch_ecs_list.append(max_repeat)
            max_ecs_needed = max(max_ecs_needed, max_repeat)
        return {
            'max_ecs_needed': max_ecs_needed
        }
