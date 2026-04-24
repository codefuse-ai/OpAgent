import re
import numpy as np
import sys
import os

def calculate_scores(log_file_path):
    format_scores = []
    format_score_thresholds = []
    answer_scores = []

    # Regex pattern to match the log line
    # Based on user query: format_score: 1.5, format_score_threshold: 0.5, answer_score: 0.5
    # Allowing for potential spaces or slight variations, and handling negative numbers just in case
    pattern = re.compile(r"format_score:\s*([-\d\.]+),\s*format_score_threshold:\s*([-\d\.]+),\s*answer_score:\s*([-\d\.]+)")

    print(f"Reading log file: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"Error: File not found at {log_file_path}")
        return

    count = 0
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    fs = float(match.group(1))
                    fst = float(match.group(2))
                    ans = float(match.group(3))
                    
                    # User requested non-negative mean
                    if fs >= 0:
                        format_scores.append(fs)
                    
                    if fst >= 0:
                        format_score_thresholds.append(fst)
                        
                    if ans >= 0:
                        answer_scores.append(ans)
                    
                    count += 1
                except ValueError:
                    continue

    print(f"Found {count} matching lines.")
    print("-" * 50)

    def print_stat(name, data):
        if data:
            mean_val = np.mean(data)
            print(f"{name}:")
            print(f"  Count (non-negative): {len(data)}")
            print(f"  Mean: {mean_val:.4f}")
        else:
            print(f"{name}: No non-negative values found.")

    print_stat("Format Score", format_scores)
    print_stat("Format Score Threshold", format_score_thresholds)
    print_stat("Answer Score", answer_scores)

if __name__ == "__main__":
    # Default path from user query
    default_log_path = "<LOG_PATH>"
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = default_log_path
        
    calculate_scores(log_path)

