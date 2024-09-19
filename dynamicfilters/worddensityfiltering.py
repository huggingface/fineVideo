import pandas as pd

#
# Given a pandas dataframe with a list of videos and the metadata extracted from YT-Commons, 
# this script creates the columns duration_seconds and word_density with the goal to study word_density across the dataset
# Finally it drops all entries in the dataframe with word density < 0.5
#

### CONFIG ###
input_pkl = 'path_to_your_input_df.pkl'
output_pkl = 'path_to_your_output_df.pkl'
visualize = False # Toggle to true to inspect some results close to 1 and 0.5 word density values.
###



df = pd.read_pickle(input_pkl)

#Adding word_density and duration_seconds to the dataframe
def duration_to_seconds(duration):
    if pd.isnull(duration):
        return 0  # or np.nan or another default
    parts = duration.split(':')
    parts = [int(p) for p in parts]
    if len(parts) == 3:  # hh:mm:ss
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:  # mm:ss
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:  # ss
        return parts[0]
    else:
        return 0  # or np.nan if format is unrecognized

# Apply the conversion function to the 'duration_string' column
df['duration_seconds'] = df['duration_string'].apply(duration_to_seconds)

# Calculate word density
# Word density is the number of words per second, so we divide word_count by duration_seconds
df['word_density'] = df.apply(lambda row: row['word_count'] / row['duration_seconds'] 
                              if row['duration_seconds'] > 0 else 0, axis=1)



if visualize:
    from tabulate import tabulate
    #Visualizing some results
    def get_samples_near_target(df, target, range_width=0.1, num_samples=3):
        """
        Get samples from the DataFrame that have 'word_density' close to the target value.

        :param df: DataFrame to sample from.
        :param target: The target word density to find samples around.
        :param range_width: The width of the range around the target value.
        :param num_samples: Number of samples to return.
        :return: A DataFrame with samples close to the target density.
        """
        # Define the range around the target
        lower_bound = target - range_width
        upper_bound = target + range_width
        
        # Filter and sample
        samples = df[(df['word_density'] >= lower_bound) & (df['word_density'] <= upper_bound)].sample(n=num_samples, random_state=1)
        return samples

    close_to_1 = get_samples_near_target(df, 1,  num_samples = 100)[['video_id', 'duration_string', 'title']]
    print(tabulate(close_to_1,headers='keys', tablefmt='pretty', showindex=False))

    close_to_05 = get_samples_near_target(df, 0.5,  num_samples = 100)[['video_id', 'duration_string', 'title']]
    print(tabulate(close_to_05,headers='keys', tablefmt='pretty', showindex=False))


# We cut at 0.5
df  = df.loc[df['word_density'] > 0.5]
print(f"Total videos: {len(df)}")
df.to_pickle(output_pkl)