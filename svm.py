# Import packages
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from PCA import find_PCs, PCA_transform
from data_processing import df, Scaling, SplitData
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from knn import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# DataSplitting(): Splits the dataset into training and testing
def DataSplitting(data):

    # Initialize the training dataset size
    train_size = 0.7

    # Call SplitData() to retrieve the train_data and test_data after splitting the dataset
    train_data, test_data = SplitData(data, train_size)

    # Obtain the X_train and y_train datasets
    X_train = train_data.drop(['id', 'diagnosis'], axis=1)
    y_train = train_data['diagnosis']

    # Obtain the X_test and y_test datasets
    X_test = test_data.drop(['id', 'diagnosis'], axis=1)
    y_test = test_data['diagnosis']

    # Return the X and y training and testing datasets
    return X_train, y_train, X_test, y_test

# LinearModel(): Obtains the linear model and plots the confusion matrix
def LinearModel(flag, X_train, y_train, X_test, y_test):

    # Define the SVM model with a linear kernel
    linear_model = SVC(kernel='linear', C=0.3)

    # Call SVM() to obtain the cross-validation results, confusion matrix, and coefficients for the linear model
    results, cm, w = SVM(linear_model, "Linear", X_train, y_train, X_test, y_test)

    # Print the coefficients
    print("The first coefficient for the SVM Linear Model is {}.".format(w[0][0]))
    print("The second coefficient for the SVM Linear Model is {}.".format(w[0][1]))

    # Plot the heatmap of the confusion matrix for the model
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=linear_model.classes_)
    disp.plot()

    # Save the model according to whether the PCA has been performed or not
    if flag == "PCA":
        plt.savefig('Results/confusion_matrix_SVM_Linear_PCA.png', dpi=300)
    else:
        plt.savefig('Results/confusion_matrix_SVM_Linear.png', dpi=300)

    # Give a line of space
    print()

    # Print the results of the cross-validation for the linear model
    print("Cross-Validation Accuracy Results For Linear Model:", results)
    print("Average Cross-Validation Accuracy For Linear Model: {:.3f}%".format(results.mean() * 100))

    # Give a line of space
    print()

    # Return the linear model and confusion matrix
    return linear_model, cm

# SVM(): Perform cross-validation on the training dataset, train the SVM model and generate confusion matrix
def SVM(model, flag, X_train, y_train, X_test, y_test):

    # Perform cross-validation to evaluate the performance of the model (5 folds)
    results = cross_val_score(model, X_train, y_train, cv=5)

    # Run the SVM model on the training dataset
    model.fit(X_train, y_train)

    # Check if the flag is linear, and if so obtain the coefficients
    if flag == "Linear":
        w = LinearCoeffs(model)

    # Generate predictions from the X_test dataset
    y_pred = model.predict(X_test)

    # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Return the cross-validation results and confusion matrix
    return results, cm, w

# LinearCoeffs(): Obtains the coefficients of the model
def LinearCoeffs(model):

    # Return the linear coefficients of the model
    return model.coef_

# Evaluate(): Find the following metrics from the confusion matrix -> accuracy, precision, recall, f1-score
def Evaluate(cm):

    # Obtain the true-negatives, false-positives, false-negatives, true-positives
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    # Calculate the accuracy, precision, recall and f1-score from the dataset
    accuracy = Accuracy(tp, tn, fp, fn)
    precision = Precision(tp, fp)
    recall = Recall(tp, fn)
    f1Score = F1Score(precision, recall)

    # Return the metrics
    return accuracy, precision, recall, f1Score

# PrecisionRecall(): Plot the precision-recall curve
def PrecisionRecall(flag, linear_model, X_test, y_test):

    # Obtain the y scores from the X test dataset
    y_scores = linear_model.decision_function(X_test)

    # Obtain the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # Graph the precision-recall curve for the model
    plt.clf()
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for SVM Classifier')

    # Save the figure
    if flag == "PCA":
        plt.savefig('Results/Precision_Recall_Curve_SVM_Linear_PCA.png', dpi=300, format='png')
    else:
        plt.savefig('Results/Precision_Recall_Curve_SVM_Linear.png', dpi=300, format='png')

# Main()
def main():

    # Give a line of space
    print()

    # Print this clarification
    print("These results are for when SVM is run on the dataset with PCA.")

    # Give a line of space
    print()

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

    # Call DataSplitting() to obtain the train and test datasets
    X_train, y_train, X_test, y_test = DataSplitting(transformed_data_df)

    # Call LinearModel() to plot the heatmap
    linear_model, cm = LinearModel("PCA", X_train, y_train, X_test, y_test)

    # Call PrecisionRecall() to plot the precision-recall curve
    PrecisionRecall("PCA", linear_model, X_test, y_test)

    # Call Evaluate() to obtain the metrics for the linear model
    accuracy, precision, recall, f1Score = Evaluate(cm)

    # Print the metrics for the linear model
    print("The Accuracy of SVM with Linear Model is: {:.3f}%.".format(accuracy))
    print("The Precision of SVM with Linear Model is: {:.3f}%.".format(precision))
    print("The Recall of SVM with Linear Model is: {:.3f}%.".format(recall))
    print("The F1-Score of SVM with Linear Model is: {:.3f}%.".format(f1Score))

    # Give 2 lines of space
    print()
    print()

    #--------------------------------------------------------------------------------------

    # Print this clarification
    print("These results are for when SVM is run on the dataset without PCA.")

    # Give a line of space
    print()

    # Call Scaling() to utilize the scaler package to standardize the data
    data = Scaling(df)

    # Reinsert the id and diagnosis columns in the dataframe
    data.insert(0, 'id', df['id'].values)
    data.insert(1, 'diagnosis', df['diagnosis'].values)

    # Initialize the training size
    train_size = 0.7

    # Split the data into training and testing dataset
    train_data, test_data = SplitData(data, train_size)

    # Obtain the X_train and y_train datasets
    X_train = train_data.drop(['id', 'diagnosis'], axis=1)
    y_train = train_data['diagnosis']

    # Obtain the X_test and y_test datasets
    X_test = test_data.drop(['id', 'diagnosis'], axis=1)
    y_test = test_data['diagnosis']

    # Call LinearModel() to plot the heatmap
    linear_model, cm = LinearModel("", X_train, y_train, X_test, y_test)

    # Call PrecisionRecall() to plot the precision-recall curve
    PrecisionRecall("", linear_model, X_test, y_test)

    # Call Evaluate() to obtain the metrics for the linear model
    accuracy, precision, recall, f1Score = Evaluate(cm)

    # Print the metrics for the linear model
    print("The Accuracy of SVM with Linear Model is: {:.3f}%.".format(accuracy))
    print("The Precision of SVM with Linear Model is: {:.3f}%.".format(precision))
    print("The Recall of SVM with Linear Model is: {:.3f}%.".format(recall))
    print("The F1-Score of SVM with Linear Model is: {:.3f}%.".format(f1Score))

    # Give a line of space
    print()

if __name__=="__main__":
    main()
