# Import packages
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from data_processing import df, Scaling, SplitData
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from KNN import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# SVM(): Perform cross-validation on the training dataset, train the SVM model and generate confusion matrix
def SVM(model, X_train, y_train, X_test, y_test):

    # Perform cross-validation to evaluate the performance of the model (5 folds)
    results = cross_val_score(model, X_train, y_train, cv=5)

    # Run the SVM model on the training dataset
    model.fit(X_train, y_train)

    # Generate predictions from the X_test dataset
    y_pred = model.predict(X_test)

    # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Return the cross-validation results and confusion matrix
    return results, cm

# Evaluate(): Find the following metrics from the confusion matrix -> accuracy, precision, recall, f1-score
def Evaluate(cm):

    # Obtain the true-negatives, false-positives, false-negatives, true-positives
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    # Obtain the accuracy, precision, recall and f1-score from the dataset
    accuracy = Accuracy(tp, tn, fp, fn)
    precision = Precision(tp, fp)
    recall = Recall(tp, fn)
    f1Score = F1Score(precision, recall)

    # Return the metrics
    return accuracy, precision, recall, f1Score

# Main()
def main():

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

    # Define the SVM model with a linear kernel
    linear_model = SVC(kernel='linear', C=0.3)

    # Call SVM() to obtain the cross-validation results and the confusion matrix for the linear model
    results, cm = SVM(linear_model, X_train, y_train, X_test, y_test)

    # Plot a heatmap of the confusion matrix for the linear model and save the figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=linear_model.classes_)
    disp.plot()
    plt.savefig('Results/confusion_matrix_SVM_Linear.png', dpi=300)

    # Give a line of space
    print()

    # Print the results of the cross-validation for the linear model
    print("Cross-Validation Accuracy Results For Linear Model:", results)
    print("Average Cross-Validation Accuracy For Linear Model: {:.3f}%".format(results.mean() * 100))

    # Give a line of space
    print()

    # Call Evaluate() to obtain the metrics for the linear model
    accuracy, precision, recall, f1Score = Evaluate(cm)

    # Print the metrics for the linear model
    print("The Accuracy of SVM with Linear Model is: {:.3f}%".format(accuracy))
    print("The Precision of SVM with Linear Model is: {:.3f}%".format(precision))
    print("The Recall of SVM with Linear Model is: {:.3f}%".format(recall))
    print("The F1-Score of SVM with Linear Model is: {:.3f}%".format(f1Score))

    # Obtain the scores for the linear model
    y_scores = linear_model.decision_function(X_test)

    # Obtain the precision and recall for the linear model
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # Graph the linear model precision-recall curve
    plt.clf()
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for SVM Classifier')
    plt.savefig('Results/Precision_Recall_Curve_SVM_Linear.png', dpi=300, format='png')

    # Give 2 lines of space
    print()
    print()

    # Initialize the polynomial model
    polynomial_model = SVC(kernel='poly')

    # Call SVM() to obtain the cross-validation results and confusion matrix
    results, cm = SVM(polynomial_model, X_train, y_train, X_test, y_test)

    # Plot a heatmap of the confusion matrix for the polynomial model and save the figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=polynomial_model.classes_)
    disp.plot()
    plt.savefig('Results/confusion_matrix_SVM_Polynomial.png', dpi=300)

    # Print the cross-validation results for the polynomial model
    print("Cross-Validation Accuracy Results For Polynomial Model:", results)
    print("Average Cross-Validation Accuracy For Polynomial Model: {:.3f}%".format(results.mean() * 100))

    # Give a line of space
    print()

    # Obtain the metrics for the confusion matrix
    accuracy, precision, recall, f1Score = Evaluate(cm)

    # Print the statistics for the polynomial model
    print("The Accuracy of SVM with Polynomial Model is: {:.3f}%".format(accuracy))
    print("The Precision of SVM with Polynomial Model is: {:.3f}%".format(precision))
    print("The Recall of SVM with Polynomial Model is: {:.3f}%".format(recall))
    print("The F1-Score of SVM with Polynomial Model is: {:.3f}%".format(f1Score))

    # Retreve the scores for the polynomial model
    y_scores = polynomial_model.decision_function(X_test)

    # Obtain the precision and recall for the polynomial model
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # Graph the polynomial model precision-recall curve
    plt.clf()
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for SVM Classifier')
    plt.savefig('Results/Precision_Recall_Curve_SVM_Polynomial.png', dpi=300, format='png')

if __name__=="__main__":
    main()
