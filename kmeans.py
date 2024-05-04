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

    # Initialize the objective value
    obj = 0

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

        # Increment the objective value
        obj += (min_dist * min_dist)

    # Return clusters
    return clusters, obj

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

# Unnormalize(): Unnormalizes the means that were obtained for each feature
def Unnormalize(means, actual_data):

    # Copy the means into a new variables
    unnormalized_means = means.copy()

    # Iterate through each feature in the means
    for feature in means.columns:

        # Ensure that the feature is not id or diagnosis
        if feature != "id" and feature != "diagnosis":

            # Multiply the means of each feature by the standard deviation and add the mean to unnormalize
            unnormalized_means[feature] = (means[feature] * actual_data.std()[feature]) + actual_data.mean()[feature]

    # Drop the id and diagnosis columns from the unnormalized means matrix
    unnormalized_means.drop(["id", "diagnosis"], axis=1, inplace=True)

    # Set the display all columns option
    pd.set_option('display.max_columns', None)

    # Return the unnormalized means
    return unnormalized_means

# Plot(): Plots the objective values obtained
def Plot(obj_vals):

    # Plotting the objective values
    plt.plot(obj_vals)

    # Adding labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Objective Value')
    plt.title('Objective Value Plot')

    # Save the plot
    plt.savefig('Results/Objective_Value_Plot.png')

# Heatmap_Raw(): Generates heatmap of the raw data
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

# Heatmap_Expression(): Generates heatmap of the expression data
def Heatmap_Expression(data, clusters, k):

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

        # Add the rows to the clusters dataFrame
        cluster_data = pd.concat([cluster_data, rows], ignore_index=True)

    # Drop the id and diagnosis columns before obtaining the correlation matrix
    cleaned_data = cluster_data.drop(['id', 'diagnosis'], axis=1, errors='ignore')

    # Obtain the correlation matrix of the expression data
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

    # Obtain the actual dataset from the file
    actual_data = pd.read_csv('data.csv')

    # Drop the last column
    actual_data.drop(actual_data.columns[-1], axis=1, inplace=True)

    # Perform one-hot encoding for the diagnosis categorical feature (B = 0, M = 1)
    actual_data["diagnosis"] = actual_data["diagnosis"].replace('B', 0)
    actual_data["diagnosis"] = actual_data["diagnosis"].replace('M', 1)

    # Normalize the dataframe
    data = Normalization(actual_data)

    # Set k = 3 to classify into 3 stages of cancer
    k = 3

    # Call Initialization() to obtain the random points
    random_points = Initialization(data, k)

    # Initialize a list for the objective values
    obj_vals = []

    # Call Clustering() to obtain the clusters
    clusters, obj = Clustering(data, random_points)

    # Initialize the old objective value to 0
    old_obj = 0

    # Plot the heatmap of the raw data
    Heatmap_Raw(data, k)

    # Implement an infinite loop
    while(True):
        # Call Means() to obtain the means
        means = Means(data, clusters)
        # Call Clustering() to obtain the clusters and objective value
        clusters, obj = Clustering(data, means)
        # Add the objective value in the list
        obj_vals.append(obj)
        # Check if objective value is equal to the old objective value
        if obj == old_obj:
            break
        # Otherwise the old objective value becomes the current objective value
        # Otherwise copy the clusters into the final clusters
        else:
            old_obj = obj
            final_clusters = copy.deepcopy(clusters)

    # Plot the heatmap of the expression data
    Heatmap_Expression(data, final_clusters, k)

    # Plot the objective values
    Plot(obj_vals)

    # Obtain the unnormalized means
    unnormalized_means = Unnormalize(means, actual_data)

    # Introduce what I am going to print
    print("These are the means for each of the clusters:")

    # Give a line of space
    print()

    # Print the unnormalized means
    print(unnormalized_means)

if __name__ == "__main__":
    main()

