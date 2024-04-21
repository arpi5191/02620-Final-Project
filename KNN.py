# Import packages
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

    correct = 0

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

    # Obtain the accuracy
    accuracy = correct/len(classifications) * 100

    # Return the metrics
    return accuracy

# Main()
def main():

    k = 3

    classifications = Clustering(trainD, testD, k)

    accuracy = Metrics(testD, classifications)

    print("The accuracy of the KNN clustering without PCA is: {:.3f}%".format(accuracy))

    #---------------------------------------------------------------------------------------------------------------------

    k = 7

    # data = Normalization(df)
    data = Scaling(df)

    num_PCs_needed, variance_covered, principle_components = find_PCs(data)
    transformed_data = PCA_transform(data, principle_components)

    column_names = [f'PC{i+1}' for i in range(num_PCs_needed)]

    transformed_data_df = pd.DataFrame(transformed_data)
    transformed_data_df.columns = column_names
    transformed_data_df.insert(0, 'id', df['id'].values)
    transformed_data_df.insert(1, 'diagnosis', df['diagnosis'].values)

    train_size = 0.7

    train_data, test_data = SplitData(transformed_data_df, train_size)

    classifications = Clustering(train_data, test_data, k)

    accuracy = Metrics(test_data, classifications)

    print("The accuracy of the KNN clustering with PCA without the packages is: {:.3f}%".format(accuracy))

    #---------------------------------------------------------------------------------------------------------------------

    k = 7

    data = Scaling(df)

    pca = PCA(n_components=8)
    pca_data = pca.fit_transform(data)

    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
    pca_df.insert(0, 'id', df['id'].values)
    pca_df.insert(1, 'diagnosis', df['diagnosis'].values)

    train_size = 0.7

    train_data, test_data = SplitData(pca_df, train_size)

    classifications = Clustering(train_data, test_data, k)

    accuracy = Metrics(test_data, classifications)

    print("The accuracy of the KNN clustering is: {:.3f}%".format(accuracy))

if __name__=="__main__":
    main()
