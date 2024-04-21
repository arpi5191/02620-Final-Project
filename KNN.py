# Import packages
import math
import pandas as pd

# SplitData(): Splits the data into training and testing subsets (70-30)
def SplitData(data, train_size):

    # Shuffle the data
    shuffled_data = data.sample(frac=1)

    # Find the index up until which the training data will be extracted
    trainInd = int(len(shuffled_data) * train_size)

    # Obtain the training and testing data
    train_data = shuffled_data[:trainInd]
    test_data = shuffled_data[trainInd:]

    # Return the training and testing data
    return train_data, test_data

# Normalization(): Normalize the data through standardization
def Normalization(data):

    # Iterate through the names and corresponding data for each feature
    for feature_name, feature_data in data.items():
        # Check if the feature is not id or diagnosis and if not apply the standardization
        if feature_name != 'id' and feature_name != 'diagnosis':
            # Find the mean of the feature
            feature_mean = data[feature_name].mean()
            # Find the standard deviation of the feature
            feature_std = data[feature_name].std()
            # Standardize the feature: Subtract the mean from the data point and divide by the standard deviation
            data[feature_name] = (data[feature_name] - feature_mean)/feature_std

    # Return the data
    return data

# Clustering(): Obtains the k-nearest neighbors
def Clustering(train_data, test_data, k):

    # Create a dictionary for neighbors
    neighbors = {}

    # Create a dictionary for classifications
    classifications = {}

    # Iterate through the indices and rows in the test data
    for test_index, test_row in test_data.iterrows():
        # Convert the features in the test data to a list
        test_features = test_row[2:].tolist()
        # Clear the neighbors list
        neighbors.clear()
        # Iterate through the indices and rows in the train data
        for train_index, train_row in train_data.iterrows():
            # Convert the features in the train data to a list
            train_features = train_row[2:].tolist()
            # Find the distance between the train and test features
            distance = Distance(train_features, test_features)
            # Extract the ID number of the training data row
            index = train_row[0]
            # Extract the classification of the training data row
            classification = train_row[1]
            # Set the distance in the neighbors dictionary
            neighbors[(index, classification)] = distance
        # Sort the neighbors dictionary by values (distance)
        sorted_neighbors = {k: v for k, v in sorted(neighbors.items(), key=lambda item: item[1])}
        # Extract the first three neighbors
        selected_neighbors = dict(list(sorted_neighbors.keys())[:k])
        # Find the number of benign classifications that have occurred
        zeroes_count = sum(1 for value in selected_neighbors.values() if value == 0)
        # Find the number of malignant classifications that have occurred
        ones_count = sum(1 for value in selected_neighbors.values() if value == 1)
        # Check if the benign classifications is greater than the number of malignant classifications
        # If so assign the label as benign, otherwise malignant
        if zeroes_count > ones_count:
            classifications[test_index] = 0
        else:
            classifications[test_index] = 1

    return classifications

# Distance(): Finds the Euclidean distance between the training and testing features
def Distance(train_features, test_features):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(train_features, test_features)))

# Metrics(): Finds the metrics to evaluate the efficiency of the classifications
def Metrics(test_data, classifications):

    correct = 0

    # Iterate through the indices and rows in the test data
    for test_index, test_row in test_data.iterrows():
        # Convert the features in the test data to a list
        test_features = test_row[2:].tolist()
        # Obtain the actual classification from the row
        real_classification = test_row[1]
        # Obtain the predicted classifications from the clustering we performed
        predicted_classification = classifications[test_index]
        # Check if the real classification is equal to the predicted classification
        # If so, increment # correct
        if real_classification == predicted_classification:
            correct += 1

    # Obtain the accuracy
    accuracy = correct/len(classifications) * 100

    # Return the metrics
    return accuracy

# Main()
def main():

    # Read the data from the csv file
    df = pd.read_csv('data.csv')

    # Drop the last row as it is meaningless
    df.drop(df.columns[-1], axis=1, inplace=True)

    # Perform one-hot encoding for the diagnosis categorical feature (B = 0, M = 1)
    df["diagnosis"] = df["diagnosis"].replace('B', 0)
    df["diagnosis"] = df["diagnosis"].replace('M', 1)

     # Normalize the dataframe
    data = Normalization(df)

    # Set the training size
    train_size = 0.7

    # Call SplitData() to get the train and test data
    train_data, test_data = SplitData(data, train_size)

    # Set the number of k-nearest neighbors
    k = 3

    # Call Clustering() to obtain the k-nearest neighbors for each datapoint in the test dataset
    classifications = Clustering(train_data, test_data, k)

    # Call Metrics() to evaluate the accuracy of the KNN clustering
    accuracy = Metrics(test_data, classifications)

    # Print the accuracy
    print("The accuracy of the KNN clustering is: {:.3f}%".format(accuracy))

if __name__=="__main__":
    main()
