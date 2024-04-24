# Import packages
import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import Normalization

# Initialization(): Randomly intializes the points for the K-Means clustering
def Initialization(data, k):

    # Obtain the centroids
    random_points = data.sample(n=k)

    # Return the points of the centroid
    return random_points

# Clustering(): Clusters the patients into multiple subtypes utilizing the means of the centroids
def Clustering(data, random_points):

    # Initialize a dictionary of clusters
    clusters = {i: [] for i in range(len(random_points))}

    # Iterate through the data
    for data_num, data_row in data.iterrows():

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
def Means(data, clusters):

    # Extract the column names
    columns = data.columns

    # Initialize the empty means dataframe
    means = pd.DataFrame(columns=columns)

    # Iterate through every key and value in the clusters
    for key, value in clusters.items():

        # Set the means['id'] to the key
        means['id'] = key

        # Initialize feature dictionary
        feature_dict = {col: 0 for col in data.columns}

        # Iterate through every index in the value
        for index in value:

            # Extract the features
            row = data.loc[data['id'] == index]

            # Iterate through the features
            for feature in row:

                # Check if the feature is valid
                # If so increment the feature in the dictionary
                if feature != 'id' and feature != 'diagnosis':
                    feature_dict[feature] += data.loc[data['id'] == index, feature].iloc[0]

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

def Heatmap_Raw(data, k):

    # Drop the id and diagnosis columns before obtaining the correlation matrix
    cleaned_data = data.drop(['id', 'diagnosis'], axis=1, errors='ignore')

    # Obtain the correlation matrix of the raw data
    corr_mat = np.corrcoef(cleaned_data, rowvar = True)

    # Plot the heatmap of the raw data
    plt.figure(figsize=(8, 8))
    plt.imshow(corr_mat, cmap='viridis', interpolation='nearest')
    plt.title("Heatmap of the Raw Data")
    plt.colorbar()
    plt.savefig("Results/Heatmap_Raw_Data_" + str(k) + "_.png")
    plt.close()

def Heatmap_Expression(data, actual_data, clusters, k):

    # Assuming data is your DataFrame and clusters is a dictionary with each cluster's IDs as lists
    columns = data.columns

    # Create an empty DataFrame for cluster values
    cluster_data = pd.DataFrame(columns=columns)

    # Define Index
    i = 1

    # Iterate through the clusters dictionary, obtain the genes in each cluster, and add the genes to the cluster_vals
    for key, value in clusters.items():

        # Extract rows where 'id' is in the list of values for this cluster
        rows = data[data['id'].isin(value)]

        # Extract actual rows where 'id' is in the list of values for this cluster
        actual_rows = actual_data[actual_data['id'].isin(value)]

        # Find the following metrics
        means = actual_rows.mean()
        medians = actual_rows.median()
        std_devs = actual_rows.std()

        # Print the mean
        print("Means for Cluster {} is:\n{}".format(i, means))

        # Give a line of space
        print()

        # Print the median
        print("Medians for Cluster {} is:\n{}".format(i, medians))

        # Give a line of space
        print()

        # Print the standard deviation
        print("Standard Deviations for Cluster {} is:\n{}".format(i, std_devs))

        # Give two lines of space
        print()
        print()

        # Add the rows to the clusters dataFrame
        cluster_data = pd.concat([cluster_data, rows], ignore_index=True)

        # Increment the Index
        i += 1

    # Drop the id and diagnosis columns before obtaining the correlation matrix
    cleaned_data = cluster_data.drop(['id', 'diagnosis'], axis=1, errors='ignore')

    # Obtain the correlation matrix of the raw data
    corr_mat = np.corrcoef(cleaned_data, rowvar = True)

    # Plot the heatmap of the clustered data
    plt.figure(figsize=(8, 8))
    plt.imshow(corr_mat, cmap='viridis', interpolation='nearest')
    plt.title("Heatmap of the Clustered Data")
    plt.colorbar()
    plt.savefig("Results/Heatmap_Clustered_Data" + str(k) + "_.png")
    plt.close()

# Main()
def main():

    # print(data)
    #
    # data = data[data["diagnosis"] != 0]

    # Obtain the actual dataset from the file
    actual_data = pd.read_csv('data.csv')

    # Drop the last row as it is meaningless
    actual_data.drop(actual_data.columns[-1], axis=1, inplace=True)

    actual_data = actual_data[actual_data["diagnosis"] != 'B']

    # Perform one-hot encoding for the diagnosis categorical feature (B = 0, M = 1)
    # actual_data["diagnosis"] = actual_data["diagnosis"].replace('B', 0)
    actual_data["diagnosis"] = actual_data["diagnosis"].replace('M', 1)

    # Read the data from the csv file
    data = pd.read_csv('data.csv')

    # Drop the last row as it is meaningless
    data.drop(data.columns[-1], axis=1, inplace=True)

    data = data[data["diagnosis"] != 'B']

    # Perform one-hot encoding for the diagnosis categorical feature (B = 0, M = 1)
    data["diagnosis"] = data["diagnosis"].replace('M', 1)

    # Normalize the dataframe
    data = Normalization(data)

    print(data)

    k = 4

    # Call Initialization to obtain the random points
    random_points = Initialization(data, k)

    # Call Clustering() to obtain the new clusters
    clusters = Clustering(data, random_points)

    # Plot the heatmap of the raw data
    Heatmap_Raw(data, k)

    # Implement an infinite loop
    while(True):
        # Call Means() to obtain the means
        means = Means(data, clusters)
        # Call Clustering() to obtain the clusters
        new_clusters = Clustering(data, means)
        # Check if clusters equal the new clusters
        if clusters == new_clusters:
            break
        # Otherwise copy the new clusters into clusters
        else:
            clusters = copy.deepcopy(new_clusters)

    # Plot the heatmap of the expression data
    Heatmap_Expression(data, actual_data, clusters, k)

if __name__ == "__main__":
    main()
