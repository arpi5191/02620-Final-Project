# Import packages
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from data_processing import df, Scaling, SplitData
from sklearn.model_selection import cross_val_score
from KNN import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Perform cross-validation to evaluate model performance
results = cross_val_score(linear_model, X_train, y_train, cv=5)
print("Cross-Validation Accuracy results:", results)
print("Average Cross-Validation Accuracy: {:.3f}%".format(results.mean() * 100))

# Give a line of space
print()

# Train the SVM model on the entire training set for final use
linear_model.fit(X_train, y_train)

# Generate predictions on the X_test dataset
y_pred = linear_model.predict(X_test)

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create and plot a confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=linear_model.classes_)
disp.plot()
plt.savefig('Images/confusion_matrix_SVM.png', dpi=300)

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

# Print the statistics
print("The Accuracy of SVM is: {:.3f}%".format(accuracy))
print("The Precision of SVM is: {:.3f}%".format(precision))
print("The Recall of SVM is: {:.3f}%".format(recall))
print("The F1-Score of SVM is: {:.3f}%".format(f1Score))
