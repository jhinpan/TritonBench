import os
import json

def convert_json_to_jsonl(source_dir):
    """
    Converts all .json files in the specified directory to .jsonl format.
    Each object in the JSON array is written as a new line in the .jsonl file.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Directory not found at {source_dir}")
        return

    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(source_dir, filename)
            jsonl_path = os.path.join(source_dir, filename.replace(".json", ".jsonl"))

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    with open(jsonl_path, 'w', encoding='utf-8') as f:
                        for item in data:
                            f.write(json.dumps(item) + '\n')
                    print(f"Successfully converted {json_path} to {jsonl_path}")
                else:
                    print(f"Skipping {json_path}: content is not a JSON array.")

            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {json_path}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {json_path}: {e}")

if __name__ == "__main__":
    data_directory = "data"
    convert_json_to_jsonl(data_directory) 