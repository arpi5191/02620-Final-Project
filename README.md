# Breast Cancer Diagnosis and Classification

Performs cancer tumor and stage classification on the Breast Cancer Wisconsin Dataset.

# Instructions:
  1) Navigate to the `02620-Final-Project` directory in the terminal.
  2) Type `python3 logistic_regression.py` in the terminal to implement logistic regression on the dataset, both before and after Principal Component Analysis.
  3) Type `python3 svm.py` in the terminal to implement Support Vector Machines on the dataset, both before and after Principal Component Analysis.
  4) Type `python3 calculate_metrics.py` in the terminal to obtain the metrics for Logistic Regression and SVM.
  5) Type `python3 generate_graphs.py` in the terminal to generate the scatterplots for the Logistic Regression and SVM.
  6) Type `python3 knn.py` in the terminal to implement K-Nearest Neighbors on the dataset, both before and after Principal Component Analysis.
  7) Type `python3 kmeans.py` in the terminal to perform KMeans clustering on the dataset.

# File Descriptions:
  1) `data.csv`: Contains the Breast Cancer Wisconsin Dataset.
  2) `data_preprocessing.py`: Splits the data into training and testing sets. Normalizes and scales each feature.
  3) `PCA.py`: Performs Principal Component Analysis on the dataset using 7 principal components. The implementation is done entirely from scratch.
  4) `logistic_regression.py`: Executes Logistic Regression on the dataset before and after performing Principal Component Analysis. The implementation is done entirely from scratch
  5) `knn.py`: Executes K-Nearest Neighbors on the dataset, both before and after performing Principal Component Analysis. The implementation is done entirely from scratch except for the scaler package that was utilized to preprocess the data in the PCA implementation only.
  6) `svm.py`: Executes Support Vector Machines on the dataset before and after performing Principal Component Analysis. The implementation utilizes packages.
  7) `kmeans.py`: Performs KMeans clustering on the dataset. The implementation is done entirely from scratch.
  8) `calculate_metrics.py`: Calculates the accuracy, precision, recall, and F1-score of the models.
  9) `generate_graphs.py`: Generates the graphs for Logistic Regression and Support Vector Machines.


# Results Directory Descriptions:
  1) `LogReg`: Logistic Regression No PC Transform.png, Logistic Regression PC Transformed.png, SVM no PC Transform.png, SVM PC Transformed.png
  2) `KNN`: confusion_matrix_KNN.png, confusion_matrix_KNN.png, KNN_Euclidean_Accuracies_PCA.png, KNN_Euclidean_Accuracies.png, KNN_Euclidean_Distance_Results.png, KNN_Manhattan_Accuracies_PCA.png, KNN_Manhattan_Accuracies.png, KNN_Manhattan_Distance_Results.png, KNN_Minkowski_Accuracies_PCA.png, KNN_Minkowski_Accuracies.png, KNN_Minkowski_Distance_Results.png
  3) `SVM`: confusion_matrix_SVM_Linear_PCA.png, confusion_matrix_SVM_Linear.png, SVM_Results.png
  4) `KMeans`: Heatmap_Raw_Data_4_.png, Heatmap_Clustered_Data4_.png, Objective_Value_Plot.png, KMeans_Results.png

# Logistic Regression Results Descriptions:
  1) `Logistic Regression No PC Transform.png`: Graph of the Logistic Regression when it is executed without PCA.
  2) `Logistic Regression PC Transformed.png`: Graph of the Logistic Regression when it is executed with PCA.
  3) `SVM No PC Transform.png`: Graph of the SVM when it is executed without PCA.
  4) `SVM PC Transformed.png`: Graph of the SVM when it is executed with PCA.


# KNN Results Descriptions:
  1) `confusion_matrix_KNN_PCA.png`: The confusion matrix of the KNN implementation with PCA.
  2) `confusion_matrix_KNN.png`: The confusion matrix of the KNN implementation without PCA.
  3) `KNN_Euclidean_Accuracies_PCA.png`: Plots the accuracies of the KNN model when Euclidean distance is utilized and PCA is implemented.
  4) `KNN_Euclidean_Accuracies.png`: Plots the accuracies of the KNN model when Euclidean distance is utilized and PCA is not implemented.
  5) `KNN_Euclidean_Distance_Results.png`: Contains a picture of the KNN results with and without PCA implementation, utilizing Euclidean distance, that we added to our final report.
  6) `KNN_Manhattan_Accuracies_PCA.png`: Plots the accuracies of the KNN model when Manhattan distance is utilized and PCA is implemented.
  7) `KNN_Manhattan_Accuracies.png`: Plots the accuracies of the KNN model when Manhattan distance is utilized and PCA is not implemented.
  8) `KNN_Manhattan_Distance_Results.png`: Contains a picture of the KNN results with and without PCA implementation, utilizing Manhattan distance, that we added to our final report.
  9) `KNN_Minkowski_Accuracies_PCA.png`: Plots the accuracies of the KNN model when Minkowski distance is utilized and PCA is implemented.
  7) `KNN_Minkowski_Accuracies.png`: Plots the accuracies of the KNN model when Minkowski distance is utilized and PCA is not implemented.
  8) `KNN_Minkowski_Distance_Results.png`: Contains a picture of the KNN results with and without PCA implementation, utilizing Minkowski distance, that we added to our final report.
     
# SVM Results Descriptions:
  1) `confusion_matrix_SVM_Linear_PCA.png`: The confusion matrix of the SVM implementation with PCA.
  2) `confusion_matrix_KNN.png`: The confusion matrix of the KNN implementation without PCA.
  3) `SVM_Results.png`: Contains a picture of the SVM results with and without PCA implementation, that we added to our final report.

# KMeans Results Descriptions:
  1) `Heatmap_Raw_Data4_.png`: The heatmap of the raw dataset.
  2) `Heatmap_Clustered_Data4_.png`: The heatmap of the dataset clustered by KNN.
  3) `KMeans.png`: Contains a picture of the KMeans results that we added to our final report.
  4) `Objective_Value_Plot.png`: Plots the objective values of the model for every iteration the KMeans algorithm runs until convergence.

