# Import packages
import copy
import math
import pandas as pd
from data_processing import df, Normalization, SplitData

# Initialization(): Randomly intializes the points for the K-Means clustering
def Initialization(df):

    # Obtain the centroids
    random_points = df.sample(n=4)

    # Return the points of the centroid
    return random_points

# Clustering(): Clusters the patients into multiple subtypes utilizing the means of the centroids
def Clustering(df, random_points):

    # Initialize a dictionary of clusters
    clusters = {i: [] for i in range(len(random_points))}

    # Iterate through the data
    for data_num, data_row in df.iterrows():

        # Obtain the index of the data and it's features
        data_index = data_row[0]
        data_features = data_row[2:].tolist()

        # Initialize the minimum and current distance
        min_dist = math.inf
        cur_dist = 0

        # Initialize the index
        i = 0

        # Iterate through the centroid numbers and rows in the random points
        for centroid_num, centroid_row in random_points.iterrows():

            # Retrieve the index and features of the centroid
            centroid_index = centroid_row[0]
            centroid_features = centroid_row[2:].tolist()

            # Obtain the current distance
            cur_dist = Euclidean_Distance(data_features, centroid_features)

            # Check if the current distance is less than the minimum distance
            # If so, set the minimum distance and cluster index
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_cluster_index = i

            # Increment the index
            i += 1

        # Add data index in the clusters
        clusters[min_cluster_index].append(data_index)

    # Return clusters
    return clusters

# Euclidean_Distance(): Finds the euclidean distance between the training and testing features
def Euclidean_Distance(train_features, test_features):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(train_features, test_features)))

# Means(): Find the means of all the features in every cluster
def Means(df, clusters):

    # Extract the column names
    columns = df.columns

    # Initialize the empty means dataframe
    means = pd.DataFrame(columns=columns)

    # Iterate through every key and value in the clusters
    for key, value in clusters.items():

        # Set the means['id'] to the key
        means['id'] = key

        # Initialize feature dictionary
        feature_dict = {col: 0 for col in df.columns}

        # Iterate through every index in the value
        for index in value:

            # Extract the features
            row = df.loc[df['id'] == index]

            # Iterate through the features
            for feature in row:

                # Check if the feature is valid
                # If so increment the feature in the dictionary
                if feature != 'id' and feature != 'diagnosis':
                    feature_dict[feature] += df.loc[df['id'] == index, feature].iloc[0]

        # Convert the feature dictionary into values
        vals = list(feature_dict.values())

        # Ensure the length of the value is not 0, before obtaining the feature values
        if len(value) != 0:
            feature_vals = [val / len(value) for val in vals]
        else:
            feature_vals = [0] * len(feature_dict)

        # Set the means to the feature values
        means.loc[key] = feature_vals

    # Return means
    return means

# Main()
def main():

    # Call Initialization to obtain the random points
    random_points = Initialization(df)

    # Call Clustering() to obtain the new clusters
    clusters = Clustering(df, random_points)

    # Implement an infinite loop
    while(True):
        # Call Means() to obtain the means
        means = Means(df, clusters)
        # Call Clustering() to obtain the clusters
        new_clusters = Clustering(df, means)
        # Check if clusters equal the new clusters
        if clusters == new_clusters:
            break
        # Otherwise copy the new clusters into clusters
        else:
            clusters = copy.deepcopy(new_clusters)

if __name__ == "__main__":
    main()
