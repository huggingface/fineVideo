import json
import os
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import sys
from math import ceil

#
# This script will run the defined prompts against one or more TGI services
# the prompts are stored in chunks in a folder called prompts/ 

# The script is called with 3 parameters:
# python tgi_inference_client.py <server_address> <port> <block_number>
# block_number is a number between 0 and 3 (both included). Those blocks are 4 subdivisions of the prompts in prompts/ 
# and by specifying the block number we run inference in each different block, this allow us to parallelize inference.
#




# Ensure the output directory exists
os.makedirs("processed", exist_ok=True)

# Function to load prompts from a single JSON file
def load_prompts_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        tasks = json.load(file)
    return tasks

# Function to process a single file's tasks and save results
def process_file(file_path, tokenizer, endpoint_url):
    # Load tasks from the current file
    tasks = load_prompts_from_file(file_path)
    results = []

    # Headers for the HTTP request
    headers = {
        "Content-Type": "application/json",
    }

    # Process each task
    for task in tqdm(tasks, desc="Processing tasks"):
        video_id = task['video_id']
        input_text = task['prompt']
        input_text = input_text.replace("Given those categories:", "Given this taxonomy:")
        pattern = r"Categories: \[.*?\]\n?"
        input_text = re.sub(pattern, '', input_text)
        pattern = r"Tags: \[.*?\]\n?"
        input_text = re.sub(pattern, '', input_text)
        pattern = r"Description: \[.*?\]\n?"
        input_text = re.sub(pattern, '', input_text)
        input_text = input_text + "RETURN A CATEGORY FROM THE TAXONOMY PROVIDED: "

        prompt_tokens = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": input_text},
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare the data for the request
        data = {
            "inputs": prompt_tokens,
            "parameters": {
                "max_new_tokens": 20,  # Adjust as needed
            },
        }

        # Make a synchronous request to the model endpoint
        response = requests.post(endpoint_url, headers=headers, json=data)
        if response.status_code == 200:
            response_data = response.json()
            completion = response_data.get('generated_text', '')
        else:
            completion = "Error: Unable to get response"

        # Append the result
        results.append({"video_id": video_id, "completion": completion})

    # Save results to file after processing all tasks in the file
    output_filename = os.path.splitext(os.path.basename(file_path))[0]
    with open(f"processed/{output_filename}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Main function to process a subset of files
def main():
    # Get server address, port, and block number from command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <server_address> <port> <block_number>")
        sys.exit(1)

    server_address = sys.argv[1]
    port = sys.argv[2]
    block_number = int(sys.argv[3])

    # Validate block number
    if block_number < 0 or block_number > 3:
        print("Error: block_number must be between 0 and 3.")
        sys.exit(1)

    # Construct endpoint URL
    endpoint_url = f"http://{server_address}:{port}/generate"

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

    # List all JSON files in the prompts directory
    files = [f for f in os.listdir("prompts") if f.endswith(".json")]

    # Sort files to ensure consistent partitioning
    files.sort()

    # Divide files into 4 blocks
    total_files = len(files)
    block_size = ceil(total_files / 4)

    # Determine start and end indices for the current block
    start_index = block_number * block_size
    end_index = min(start_index + block_size, total_files)

    # Process only the files in the current block
    for i in range(start_index, end_index):
        file_path = os.path.join("prompts", files[i])
        process_file(file_path, tokenizer, endpoint_url)

# Run the main function
if __name__ == "__main__":
    main()