import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


#
# Given a pandas dataframe with a list of videos, this script will generate custom prompts for your videos and by default store
# them in a subfolder 'prompts' 
#

### CONFIG ###
df_path = 'current_videos.pkl'
###



# prompt_template = """
# Given those categories: {leaves}
# Classify a youtube video given its closed captioning and some metadata details. RETURN ONLY the selected category and nothing else!
# Title: {title}
# Description: {description}
# Categories: {categories}
# Tags: {tags}
# Channel: {channel}
# Closed Caption: {closed_caption}
# """
prompt_template = """
Given those categories: {leaves}
Classify a youtube video given its closed captioning and some metadata details. RETURN ONLY the selected category and nothing else!
Title: {title}
Description: {description}
Channel: {channel}
Closed Caption: {closed_caption}
"""

def get_leaves(taxonomy):
    leaves = []
    for key, value in taxonomy.items():
        if isinstance(value, dict) and value:  # If it's a non-empty dictionary
            leaves.extend(get_leaves(value))
        else:  # If it's an empty dictionary, consider it as a leaf
            if not value:  # Check if the value is an empty dictionary
                leaves.append(key)
    return leaves

def generate_prompt(row, text, leaves):
    return prompt_template.format(
        leaves=json.dumps(leaves, indent=2),
        title=row['title'],
        # description=row['description'],
        # categories=row['categories'],
        tags=row['tags'],
        channel=row['channel'],
        closed_caption=row['text'][:5000]  # Trim closed captions
    )

def save_prompts_to_file(prompts, output_file):
    """Save prompts to the output JSON file, overwriting it."""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(prompts, file, indent=4, ensure_ascii=False)

def process_row(row, leaves):
    video_id = row['video_id']

    # Generate the prompt
    prompt = generate_prompt(row, leaves)
    return {"video_id": video_id, "prompt": prompt}

def generate_prompts_and_save(df_path, output_dir='prompts', max_workers=None, chunksize=1000):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the taxonomy content
    with open('content_taxonomy.json', 'r') as file:
        taxonomy_content = json.load(file)

    leaves = get_leaves(taxonomy_content)

    # Load the entire DataFrame first (ensure this fits in memory)
    df = pd.read_pickle(df_path)
    
    # Process in chunks
    chunk_index = 0
    for start in range(0, len(df), chunksize):
        chunk = df.iloc[start:start + chunksize]
        prompts = []

        # Use ThreadPoolExecutor for file I/O-bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                process_row,
                (row for _, row in chunk.iterrows()),
                [leaves] * len(chunk)
            )

        # Collect results and filter out None results
        results = [result for result in results if result is not None]

        # Save results to file in chunks
        if results:
            chunk_file = os.path.join(output_dir, f'prompts_{chunk_index}.json')
            save_prompts_to_file(results, chunk_file)
            print(f"Saved chunk {chunk_index} to {chunk_file}")
            chunk_index += 1

    print(f"Completed processing.")

# Specify the number of workers, for example, 8
generate_prompts_and_save(df_path, max_workers=8)
