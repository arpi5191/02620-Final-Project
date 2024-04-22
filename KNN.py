# Import packages
import math
import numpy as np
import pandas as pd
from PCA import find_PCs, PCA_transform
from data_processing import df, trainD, testD, Scaling, Normalization, SplitData

# Clustering(): Obtains the k-nearest neighbors
def Clustering(trainD, testD, k):

    # Create a dictionary for neighbors
    neighbors = {}

    # Create a dictionary for classifications
    classifications = {}

    # Iterate through the indices and rows in the test data
    for test_index, test_row in testD.iterrows():
        # Convert the features in the test data to a list
        test_features = test_row[2:].tolist()
        # Clear the neighbors list
        neighbors.clear()
        # Iterate through the indices and rows in the train data
        for train_index, train_row in trainD.iterrows():
            # Convert the features in the train data to a list
            train_features = train_row[2:].tolist()
            # Find the distance between the train and test features
            distance = Distance(train_features, test_features)
            # Extract the ID number of the training data row
            index = train_row['id']
            # Extract the classification of the training data row
            classification = train_row['diagnosis']
            # Set the distance in the neighbors dictionary
            neighbors[(index, classification)] = distance
        # Sort the neighbors dictionary by values (distance)
        sorted_neighbors = {k: v for k, v in sorted(neighbors.items(), key=lambda item: item[1])}
        # Extract the first three neighbors
        selected_neighbors = dict(list(sorted_neighbors.keys())[:k])
        # Find the number of benign classifications that have occurred
        zeroes_count = sum(1 for value in selected_neighbors.values() if value == float(0))
        # Find the number of malignant classifications that have occurred
        ones_count = sum(1 for value in selected_neighbors.values() if value == float(1))
        # Check if the benign classifications is greater than the number of malignant classifications
        # If so assign the label as benign, otherwise malignant
        if zeroes_count > ones_count:
            classifications[test_row[0]] = 0
        else:
            classifications[test_row[0]] = 1

    return classifications

# Distance(): Finds the Euclidean distance between the training and testing features
def Distance(train_features, test_features):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(train_features, test_features)))

# Metrics(): Finds the metrics to evaluate the efficiency of the classifications
def Metrics(testD, classifications):

    # Initialize the number of correct values
    correct = 0

    # Initialize the counts
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Iterate through the indices and rows in the test data
    for test_index, test_row in testD.iterrows():
        test_id = test_row[0]
        # Convert the features in the test data to a list
        test_features = test_row[2:].tolist()
        # Obtain the actual classification from the row
        real_classification = test_row[1]
        # Obtain the predicted classifications from the clustering we performed
        predicted_classification = classifications[test_id]
        # Check if the real classification is equal to the predicted classification
        # If so, increment # correct
        if real_classification == predicted_classification:
            correct += 1
        # Find the true positives
        if real_classification == 1 and predicted_classification == 1:
            tp += 1
        # Find the true negatives
        if real_classification == 0 and predicted_classification == 0:
            tn += 1
        # Find the false positives
        if real_classification == 0 and predicted_classification == 1:
            fp += 1
        # Find the false negatives
        if real_classification == 1 and predicted_classification == 0:
            fn += 1

    # Calculate the metrics
    accuracy = Accuracy(tp, tn, fp, fn)
    precision = Precision(tp, fp)
    recall = Recall(tp, fn)
    f1Score = F1Score(precision, recall)

    # Return the metrics
    return accuracy, precision, recall, f1Score

def Accuracy(tp, tn, fp, fn):

    return (tp + tn)/(tp + tn + fp + fn) * 100

def Precision(tp, fp):

    return tp/(tp + fp) * 100

def Recall(tp, fn):

    return tp/(tp + fn) * 100

def F1Score(precision, recall):

    return (2 * precision * recall)/(precision + recall)

# Main()
def main():

    # This portion of the main() executes the KNN without the PCA

    # Initialize the k-value
    k = 3

    # Call Clustering() to obtain the classifications
    classifications = Clustering(trainD, testD, k)

    # Call Metrics() to obtain the KNN evaluations/statistics
    accuracy, precision, recall, f1Score = Metrics(testD, classifications)

    # Print the metrics
    print("The accuracy of the KNN clustering without PCA is: {:.3f}%".format(accuracy))
    print("The precision of the KNN clustering without PCA is: {:.3f}%".format(precision))
    print("The recall of the KNN clustering without PCA is: {:.3f}%".format(recall))
    print("The f1 score of the KNN clustering without PCA is: {:.3f}%".format(f1Score))

    # Give a line of space
    print()

    #---------------------------------------------------------------------------------------------------------------------

    # This portion of the main() executes the KNN with the PCA

    # Initialize the k-value
    k = 5

    # data = Normalization(df)

    # Call Scaling() to utilize the scaler() package to scale the data
    data = Scaling(df)

    # Call find_PCs() to obtain the PCs
    # Call PCA_transform() to obtain the projections of the data with the PCs
    num_PCs_needed, variance_covered, principle_components = find_PCs(data)
    transformed_data = PCA_transform(data, principle_components)

    # Obtain the names of the columns
    column_names = [f'PC{i+1}' for i in range(num_PCs_needed)]

    # Convert the transformed data into a dataframe
    # Insert the id and diagnosis columns in the dataframe
    transformed_data_df = pd.DataFrame(transformed_data)
    transformed_data_df.columns = column_names
    transformed_data_df.insert(0, 'id', df['id'].values)
    transformed_data_df.insert(1, 'diagnosis', df['diagnosis'].values)

    # Initialize the size of the training dataset
    train_size = 0.7

    # Call SplitData() to obtain the train_data and test_data after splitting the dataset
    train_data, test_data = SplitData(transformed_data_df, train_size)

    # Call Clustering() to retrieve the classifications of the data
    classifications = Clustering(train_data, test_data, k)

    # Call Metrics() to get the evaluations/statistics of the data
    accuracy, precision, recall, f1Score = Metrics(test_data, classifications)

    # Print the metrics of the data
    print("The accuracy of the KNN clustering with PCA: {:.3f}%".format(accuracy))
    print("The precision of the KNN clustering with PCA: {:.3f}%".format(precision))
    print("The recall of the KNN clustering with PCA: {:.3f}%".format(recall))
    print("The f1 score of the KNN clustering with PCA: {:.3f}%".format(f1Score))

if __name__=="__main__":
    main()
