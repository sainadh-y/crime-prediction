import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

def process_crime_data(input_file='Train.csv', output_file='crime_rate_summary.csv'):
    # Read the dataset
    df = pd.read_csv(input_file)

    # Remove Latitude and Longitude columns
    df = df.drop(columns=['Latitude', 'Longitude'])

    # Prepare a list to collect clustered dataframes
    clustered_dfs = []

    # Cluster X, Y within each HUNDRED_BLOCK using DBSCAN
    for block, group in df.groupby('HUNDRED_BLOCK'):
        coords = group[['X', 'Y']].values
        if len(coords) > 1:
            clustering = DBSCAN(eps=50, min_samples=1).fit(coords)
            group = group.copy()
            group['cluster'] = clustering.labels_
        else:
            group = group.copy()
            group['cluster'] = 0  # Single point cluster
        clustered_dfs.append(group)

    # Concatenate all clustered groups
    clustered_df = pd.concat(clustered_dfs, ignore_index=True)

    # Group by the specified columns except Latitude and Longitude, include cluster
    group_cols = ['TYPE', 'HUNDRED_BLOCK', 'NEIGHBOURHOOD', 'X', 'Y', 'YEAR', 'MONTH', 'cluster']
    crime_rate_df = clustered_df.groupby(group_cols).size().reset_index(name='crime_rate')

    # Save the summarized dataset to a new CSV file
    crime_rate_df.to_csv(output_file, index=False)
    print(f"Crime rate summary saved to {output_file}")

if __name__ == "__main__":
    process_crime_data()
