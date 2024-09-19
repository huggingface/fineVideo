import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

#
# Given a pandas dataframe with a list of videos and the metadata extracted from YT-Commons, 
# this script creates a new dataframe with a list of videoids that the target the hours of video that we want to collect.
#

### CONFIG ###
input_pkl = 'path_to_your_current_videos_df.pkl'
output_pkl = 'path_to_your_output_df.pkl'
taxonomy_path = 'content_taxonomy.json'
target_hours = 4500
###

# Step 1: Preprocess the Data
def preprocess_df(df):
    # Fill NaNs with 0 or suitable values
    df['comment_count'] = df['comment_count'].fillna(0)
    df['view_count'] = df['view_count'].fillna(0)
    df['like_count'] = df['like_count'].fillna(0)
    df['channel_follower_count'] = df['channel_follower_count'].fillna(0)
    df['duration_seconds'] = df['duration_seconds'].fillna(0)
    
    # Normalize numerical columns for fair weighting
    scaler = MinMaxScaler()
    df[['comment_count', 'view_count', 'like_count']] = scaler.fit_transform(
        df[['comment_count', 'view_count', 'like_count']]
    )
    
    return df

# Step 2: Compute User Activity Score
def compute_user_activity(df, weights=(0.2, 0.5, 0.3)):
    # Weights: 0.2 for comments, 0.5 for views, 0.3 for likes
    df['user_activity_score'] = (
        weights[0] * df['comment_count'] +
        weights[1] * df['view_count'] +
        weights[2] * df['like_count']
    )
    return df

# Step 3: Map Inferred Categories to Higher Taxonomy Levels
# Note: this was not used in the final version of the content selection algorithm but is useful data that we let in the dataset.
def map_to_parent_categories(df, taxonomy):
    """
    Maps each inferred category in the DataFrame to its top-level parent category
    in the hierarchical taxonomy.

    :param df: DataFrame containing video data with an 'inferred_category' column.
    :param taxonomy: A nested dictionary representing the hierarchical taxonomy.
    :return: DataFrame with an added 'parent_category' column representing the top-level parent category.
    """
    
    # Helper function to find the top-level parent category
    def find_top_parent_category(leaf_name, taxonomy):
        """
        Finds the top-level parent category of a given leaf in the hierarchical taxonomy.

        :param leaf_name: A string representing the leaf node to search for.
        :param taxonomy: A dictionary representing the full hierarchical taxonomy.
        :return: The top-level parent category of the given leaf if found, else None.
        """
        def recursive_search(taxonomy, leaf_name, current_top_category):
            for category, subcategories in taxonomy.items():
                if category == leaf_name:
                    # Found the leaf node; return the top-level category
                    return current_top_category
                if isinstance(subcategories, dict):
                    # Continue searching deeper
                    found_category = recursive_search(subcategories, leaf_name, current_top_category)
                    if found_category:
                        return found_category
            return None

        # Start the search with top-level categories
        for top_category, subcategories in taxonomy.items():
            result = recursive_search(subcategories, leaf_name, top_category)
            if result:
                return result

        return None

    # Map each inferred category to its top-level parent category
    df['parent_category'] = df['inferred_category'].apply(lambda x: find_top_parent_category(x, taxonomy))
    
    return df


# Step 4: Select Videos for Diversity and Total Duration
def select_videos(df, target_hours=4500):
    target_seconds = target_hours * 3600  # Convert hours to seconds
    selected_videos = pd.DataFrame()

    # Calculate the total number of inferred categories
    inferred_categories = df['inferred_category'].unique()
    total_categories = len(inferred_categories)
    
    # Calculate the initial target seconds per inferred category
    target_seconds_per_category = target_seconds / total_categories
    
    # Shuffle rows to mix categories and channels
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Initialize dictionary to keep track of selected durations per inferred category
    category_durations = {category: 0 for category in inferred_categories}
    
    # Define a progressive penalty for repeated channels
    channel_penalty_increment = 0.1  # Incremental penalty for each additional video from the same channel
    
    # Process each inferred category
    for inferred_category in inferred_categories:
        category_df = df[df['inferred_category'] == inferred_category]
        
        # Sort by user activity score and channel follower count in reverse order
        category_df = category_df.sort_values(
            by=['user_activity_score', 'channel_follower_count'],
            ascending=[False, True]
        )
        
        current_duration = 0
        channel_counter = {}
        
        for _, row in category_df.iterrows():
            if current_duration >= target_seconds_per_category:
                break
            
            channel = row['channel']
            
            # Calculate the penalty based on the number of videos already selected from this channel
            penalty_factor = 1 - (channel_counter.get(channel, 0) * channel_penalty_increment)
            penalty_factor = max(penalty_factor, 0)  # Ensure penalty factor doesn't go negative
            
            # Apply penalty by using a probability check
            if np.random.rand() < penalty_factor:
                selected_videos = pd.concat([selected_videos, pd.DataFrame([row])])
                current_duration += row['duration_seconds']
                category_durations[inferred_category] += row['duration_seconds']
                channel_counter[channel] = channel_counter.get(channel, 0) + 1
        
        # Update target duration if some categories can't meet the target
        remaining_seconds = target_seconds - selected_videos['duration_seconds'].sum()
        remaining_categories = total_categories - len(selected_videos['inferred_category'].unique())
        if remaining_categories > 0:
            target_seconds_per_category = remaining_seconds / remaining_categories
    
    # Adjust to match exactly the target duration or close
    selected_videos = selected_videos.sort_values(by='duration_seconds', ascending=True)
    
    final_selected = pd.DataFrame()
    total_duration = 0
    
    for _, row in selected_videos.iterrows():
        if total_duration + row['duration_seconds'] <= target_seconds:
            final_selected = pd.concat([final_selected, pd.DataFrame([row])])
            total_duration += row['duration_seconds']
    
    return final_selected

def main_algorithm(df, taxonomy_file, target_hours = 4500):
    df = preprocess_df(df)
    df = compute_user_activity(df)
    
    # Load taxonomy from JSON file
    with open(taxonomy_file, 'r') as file:
        taxonomy = json.load(file)
    
    # Map inferred categories to their parent categories
    df = map_to_parent_categories(df, taxonomy)
    
    # Select videos based on updated criteria
    selected_videos = select_videos(df, target_hours=target_hours)
    
    print(f"Total selected videos: {len(selected_videos)}")
    print(f"Total duration (seconds): {selected_videos['duration_seconds'].sum()}")
    
    return selected_videos

# Run the algorithm
df = pd.read_pickle(input_pkl)
selected_videos_df = main_algorithm(df, taxonomy_path, target_hours=target_hours)
select_videos.to_pickle(output_pkl)