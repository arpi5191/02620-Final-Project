# Import packages
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from data_processing import df, Scaling, SplitData
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from KNN import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def SVM(model, X_train, y_train, X_test, y_test):

    # Perform cross-validation to evaluate model performance
    results = cross_val_score(model, X_train, y_train, cv=5)

    # Train the SVM model on the entire training set for final use
    model.fit(X_train, y_train)

    # Generate predictions on the X_test dataset
    y_pred = model.predict(X_test)

    # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return results, cm

def Evaluate(cm):

    # Retrieve the following params
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    # Obtain the metrics
    accuracy = Accuracy(tp, tn, fp, fn)
    precision = Precision(tp, fp)
    recall = Recall(tp, fn)
    f1Score = F1Score(precision, recall)

    return accuracy, precision, recall, f1Score

def main():

    # Call Scaling() to utilize the scaler() package to scale the data
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

    results, cm = SVM(linear_model, X_train, y_train, X_test, y_test)

    # Create and plot a confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=linear_model.classes_)
    disp.plot()
    plt.savefig('Results/confusion_matrix_SVM_Linear.png', dpi=300)

    print()

    print("Cross-Validation Accuracy Results For Linear Model:", results)
    print("Average Cross-Validation Accuracy For Linear Model: {:.3f}%".format(results.mean() * 100))

    # Give a line of space
    print()

    accuracy, precision, recall, f1Score = Evaluate(cm)

    # Print the statistics
    print("The Accuracy of SVM with Linear Model is: {:.3f}%".format(accuracy))
    print("The Precision of SVM with Linear Model is: {:.3f}%".format(precision))
    print("The Recall of SVM with Linear Model is: {:.3f}%".format(recall))
    print("The F1-Score of SVM with Linear Model is: {:.3f}%".format(f1Score))

    y_scores = linear_model.decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    plt.clf()
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for SVM Classifier')

    plt.savefig('Results/Precision_Recall_Curve_SVM_Linear.png', dpi=300, format='png')


    print()
    print()

    polynomial_model = SVC(kernel='poly')

    results, cm = SVM(polynomial_model, X_train, y_train, X_test, y_test)

    # Create and plot a confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=polynomial_model.classes_)
    disp.plot()
    plt.savefig('Results/confusion_matrix_SVM_Polynomial.png', dpi=300)

    print("Cross-Validation Accuracy Results For Polynomial Model:", results)
    print("Average Cross-Validation Accuracy For Polynomial Model: {:.3f}%".format(results.mean() * 100))

    # Give a line of space
    print()

    accuracy, precision, recall, f1Score = Evaluate(cm)

    # Print the statistics
    print("The Accuracy of SVM with Polynomial Model is: {:.3f}%".format(accuracy))
    print("The Precision of SVM with Polynomial Model is: {:.3f}%".format(precision))
    print("The Recall of SVM with Polynomial Model is: {:.3f}%".format(recall))
    print("The F1-Score of SVM with Polynomial Model is: {:.3f}%".format(f1Score))

    y_scores = polynomial_model.decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    plt.clf()
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for SVM Classifier')

    plt.savefig('Results/Precision_Recall_Curve_SVM_Polynomial.png', dpi=300, format='png')

if __name__=="__main__":
    main()
