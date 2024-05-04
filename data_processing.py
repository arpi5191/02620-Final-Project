# Import packages
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler

# SplitData(): Splits the data into training and testing subsets (70-30)
def SplitData(data, train_size):

    # Shuffle the data
    shuffled_data = data.sample(frac=1, random_state=3).reset_index(drop=True)

    # Find the index up until which the training data will be extracted
    trainInd = int(len(shuffled_data) * train_size)

    # Obtain the training and testing data
    trainData = shuffled_data[:trainInd]
    testData = shuffled_data[trainInd:]

    # Return the training and testing data
    return trainData, testData

# Scaling(): Normalize the data with packages through standardization
def Scaling(data):

    # Initialize the StandardScaler() function
    scaler = StandardScaler()

    # Retrieve the features and scale them
    features = data.drop(['diagnosis', 'id'], axis=1)
    scaled_features = scaler.fit_transform(features)

    # Transform the scaled features into the dataframe
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Return the scaled dataframe
    return scaled_df

# Normalization(): Normalize the data without packages through standardization
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
trainD, testD = SplitData(data, train_size)
