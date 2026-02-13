import os
import json
import argparse

def calculate_accuracy(log_dir):
    total_count = 0
    success_count = 0
    
    print(f"Searching for trajectory.json files in: {log_dir}")
    
    trajectory_files = []
    for root, dirs, files in os.walk(log_dir):
        if "trajectory.json" in files:
            trajectory_files.append(os.path.join(root, "trajectory.json"))
            
    trajectory_files.sort() # Sort for consistent output
    
    for file_path in trajectory_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"[WARNING] Empty file: {file_path}")
                    continue
                data = json.loads(content)
                
            score = 0.0
            found_eval = False
            
            # Iterate through the list to find the evaluation item
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("type") == "evaluation":
                        score = item.get("score", 0.0)
                        found_eval = True
                        break
            
            if found_eval:
                total_count += 1
                if score == 1.0:
                    success_count += 1
                else:
                    print(file_path)
            else:
                print(f"[WARNING] No evaluation result found in: {file_path}")
                
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error in {file_path}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")

    if total_count > 0:
        accuracy = success_count / total_count
        print(f"\nTotal tasks: {total_count}")
        print(f"Success tasks: {success_count}")
        print(f"Accuracy: {accuracy:.2%}")
    else:
        print("\nNo valid trajectory files found.")

if __name__ == "__main__":
    default_log_dir = "/ossfs/workspace/oagent_training_onlineRL/Agent-R1/eval/demo/log/local_webarena_eval_name"
    parser = argparse.ArgumentParser(description="Calculate success rate from trajectory.json files.")
    parser.add_argument("log_dir", nargs="?", default=default_log_dir, help="Root directory of the logs")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"Error: Directory {args.log_dir} does not exist.")
    else:
        calculate_accuracy(args.log_dir)

