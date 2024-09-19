from huggingface_hub import snapshot_download
import pandas as pd
import pyarrow.parquet as pq
import os

#
# This script downloads YTCommons dataset from Hugging Face and parses some relevant fields of each video to finally store them in a dataframe
# Be careful - this script requires a decent amount of RAM to work.
#


### CONFIG ###
dataset_path = './Youtube-Commons/'
output_pkl = 'en_ycommons.pkl'
###



def read_filtered_parquet_files(folder_path, fields, filters=None):
    """
    Reads specified fields from all Parquet files in a folder with filtering and combines them into a single DataFrame.
    
    Parameters:
    folder_path (str): The path to the folder containing Parquet files.
    fields (list): List of fields to read from the Parquet files.
    filters (list): List of tuples for filtering, e.g., [('column_name', '==', value)]
    
    Returns:
    pd.DataFrame: A DataFrame containing the specified fields from all filtered Parquet files.
    """
    # List to store DataFrames
    dataframes = []
    
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")
            
            # Read the entire Parquet file
            df = pq.read_table(file_path).to_pandas()
            
            # Apply filters if provided
            if filters:
                for column, operator, value in filters:
                    if operator == '==':
                        df = df[df[column] == value]
                    elif operator == '>':
                        df = df[df[column] > value]
                    elif operator == '<':
                        df = df[df[column] < value]
                    # Add other operators as needed
            
            # Check if 'word_count' column exists and filter rows with word_count > 50
            if 'word_count' in df.columns:
                df = df[df['word_count'] > 50]
                
            # Handle 'source_language' and 'language_id_method' fields
            if 'source_language' not in df.columns and 'language_id_method' in df.columns:
                df['source_language'] = df['language_id_method']
            elif 'source_language' in df.columns:
                pass  # 'source_language' already exists, no action needed
            
            # Ensure 'source_language' is in the fields to select
            if 'source_language' not in fields:
                fields.append('source_language')
                
            # Select only the specified fields
            df = df[fields]
            dataframes.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


fields = ['acodec', 'age_limit', 'categories', 'channel', 'channel_follower_count', 'channel_id', 'character_count', 'comment_count', 'date', 'description', 'duration_string', 'language', 'license', 'like_count', 'original_language', 'resolution', 'tags', 'text', 'title', 'transcription_language', 'upload_date', 'vcodec', 'video_id', 'video_link', 'view_count', 'word_count']
filters = [('original_language', '==', 'en'), ('transcription_language', '==', 'en')]

folder = snapshot_download("PleIAs/YouTube-Commons",
                           repo_type='dataset',
                           local_dir=dataset_path)


df = read_filtered_parquet_files(dataset_path, fields, filters=filters)

print(df.head())
print(f"Total videos: {len(df)}")
df.to_pickle(output_pkl)