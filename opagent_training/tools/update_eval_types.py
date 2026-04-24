import os
import json
import shutil

def update_eval_types(source_dir, dest_dir):
    """
    Reads JSON files from source_dir, updates 'eval_types' to ["webjudge"],
    and saves them to dest_dir.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return

    if not os.path.exists(dest_dir):
        print(f"Creating destination directory: {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)

    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {source_dir}")

    updated_count = 0
    error_count = 0

    for filename in json_files:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if 'eval' key exists and update 'eval_types'
            if 'eval' in data:
                data['eval']['eval_types'] = ["webjudge"]
                
                # Write to destination file
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                updated_count += 1
            else:
                # If no eval key, just copy the file? Or skip?
                # Assuming we want to keep all files, just modify relevant ones. 
                # If 'eval' is missing, maybe we should just copy the original content.
                # However, the previous logic skipped it. I'll copy it to be safe so the dest folder is a complete set.
                print(f"Warning: {filename}: 'eval' key not found. Copying original file.")
                shutil.copy2(source_path, dest_path)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1

    print(f"Process complete. Files saved to {dest_dir}")
    print(f"Updated: {updated_count}, Errors: {error_count}")

if __name__ == "__main__":
    source_directory = "<SOURCE_EVAL_CONFIG_DIR>"
    dest_directory = "<SOURCE_EVAL_CONFIG_DIR>_webjudge"
    update_eval_types(source_directory, dest_directory)
