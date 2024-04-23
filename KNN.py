# Import packages
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PCA import find_PCs, PCA_transform
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_processing import df, Scaling, SplitData

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
            # Call Euclidean_Distance() to find the euclidean distance between the train and test features
            distance = Euclidean_Distance(train_features, test_features)
            # Call Manhattan_Distance() to find the manhattan distance between the train and test features
            # distance = Manhattan_Distance(train_features, test_features)
            # Call Manhattan_Distance() to find the manhattan distance between the train and test features
            # distance = Minkowski_Distance(train_features, test_features)
            # Extract the ID number of the training data row
            index = train_row['id']
            # Extract the classification of the training data row
            classification = train_row['diagnosis']
            # Set the Euclidean_Distance in the neighbors dictionary
            neighbors[(index, classification)] = distance
        # Sort the neighbors dictionary by values (Euclidean_Distance)
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

    # Return classifications
    return classifications

# Euclidean_Distance(): Finds the Euclidean distance between the training and testing features
def Euclidean_Distance(train_features, test_features):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(train_features, test_features)))

# Manhattan_Distance(): Finds the Manhattan distance between the training and testing features
def Manhattan_Distance(train_features, test_features):
    return sum(abs(x - y) for x, y in zip(train_features, test_features))

# Minkowski_Distance(): Finds the Minkowski distance between the training and testing features
def Minkowski_Distance(train_features, test_features, p=2):
    return math.pow(sum(abs(x - y)**p for x, y in zip(train_features, test_features)), 1/p)

# Metrics(): Finds the metrics to evaluate the efficiency of the classifications
def Metrics(flag, test_data, classifications):

    # Initialize the number of correct values
    correct = 0

    # Initialize the counts
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Iterate through the indices and rows in the test data
    for test_index, test_row in test_data.iterrows():
        # Obtain the test id
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

    # Retrieve and plot the confusion matrix
    Confusion_Matrix(flag, tn, fp, fn, tp)

    # Calculate the metrics
    accuracy = Accuracy(tp, tn, fp, fn)
    precision = Precision(tp, fp)
    recall = Recall(tp, fn)
    f1Score = F1Score(precision, recall)

    # Return the metrics
    return accuracy, precision, recall, f1Score

# ConfusionMatrix(): Obtain and graph the confusion matrix
def Confusion_Matrix(flag, tp, tn, fp, fn):

    # Retrieve the confusion matrix
    cm = np.array([[tp, fp], [tn, fn]])

    # Create and plot a confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Set the title of the plot according to whether PCA was utilized in the implementation or not
    if flag == "PCA":
        plt.savefig('Results/confusion_matrix_KNN_PCA.png', dpi=300)
    else:
        plt.savefig('Results/confusion_matrix_KNN.png', dpi=300)

# Accuracy(): Returns the accuracy
def Accuracy(tp, tn, fp, fn):

    return (tp + tn)/(tp + tn + fp + fn) * 100

# Precision(): Returns the precision
def Precision(tp, fp):

    return tp/(tp + fp) * 100

# Recall(): Returns the recall
def Recall(tp, fn):

    return tp/(tp + fn) * 100

# F1Score(): Recall the F1Score
def F1Score(precision, recall):

    return (2 * precision * recall)/(precision + recall)

# KNN(): Calculate the KNN on the train and test datasets
def KNN(flag, train_data, test_data):

    # Create a list to store the accuracies
    accuracies = []

    # Iterate through k-values from 1 to 5
    for k in range(1, 6):

        # Call Clustering() to obtain the classifications
        classifications = Clustering(train_data, test_data, k)

        # Call Metrics() to obtain the KNN evaluations/statistics
        if flag == "PCA":
            accuracy, precision, recall, f1Score = Metrics("PCA", test_data, classifications)
        else:
            accuracy, precision, recall, f1Score = Metrics("", test_data, classifications)

        # Store the accuracy
        accuracies.append(accuracy)

        # Print the accuracy, precision, recall and f1-score
        print("The accuracy of the KNN clustering {:.3f}% for k = {}.".format(accuracy, k))
        print("The precision of the KNN clustering {:.3f}% for k = {}.".format(precision, k))
        print("The recall of the KNN clustering {:.3f}% for k = {}.".format(recall, k))
        print("The f1 score of the KNN clustering {:.3f}% for k = {}.".format(f1Score, k))

        # Print an empty line
        print()

    # Obtain the k-values
    k_vals = range(1, len(accuracies) + 1)

    # Plot the accuracies over the k-values
    plt.clf()
    plt.plot(k_vals, accuracies)
    plt.title('Accuracies Over the K-Values')
    plt.xlabel('K-Values')
    plt.ylabel('Accuracies')

    # Set the title of the image
    if flag == "PCA":
        plt.savefig('Results/KNN_Euclidean_Accuracies_PCA', dpi=300)
        # plt.savefig('Results/KNN_Manhattan_Accuracies_PCA', dpi=300)
        # plt.savefig('Results/KNN_Minkowski_Accuracies_PCA', dpi=300)
    else:
        plt.savefig('Results/KNN_Euclidean_Accuracies', dpi=300)
        # plt.savefig('Results/KNN_Manhattan_Accuracies', dpi=300)
        # plt.savefig('Results/KNN_Minkowski_Accuracies', dpi=300)

# Main()
def main():

    # Give a line of space
    print()

    # Print this clarification
    print("These results are for when KNN is run on the dataset with PCA.")

    # Give a line of space
    print()

    # Initialize the k-value
    k = 5

    # Call Scaling() to utilize the scaler() package to scale the data
    data = Scaling(df)
    # data = Normalization(df)

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

    # Call KNN() to perform the KNN on the 8 PCs
    KNN("PCA", train_data, test_data)

    # Give two lines of space
    print()
    print()

    #---------------------------------------------------------------------------------------------------------------------

    # Give a line of space
    print()

    # Print this clarification
    print("These results are for when PCA is run on the dataset without PCA.")

    # Give a line of space
    print()

    # Call SplitData() to obtain the train_data and test_data after splitting the dataset
    train_data, test_data = SplitData(df, train_size)

    # Call KNN() to perform the KNN on the training and testing dataset
    KNN("", train_data, test_data)

if __name__=="__main__":
    main()
