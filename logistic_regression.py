from data_processing import trainD, testD
import numpy as np
from PCA import find_PCs, PCA_transform


training_data = trainD.to_numpy()
testing_data = testD.to_numpy()


def process_array(input_array: np.array) -> (np.array, np.array, np.array):
    # This function split the given training and testing sets into feature values and labels.
    # Inputs:
    # input_array: a numpy array containing sample id, feature values, and labels for all samples.
    # Outputs:
    # X:           a numpy array containing all the feature values for each sample.
    # Y:           a numpy array containing all the true labels.
    # id:          a numpy array containing the sample ids.

    n_rows, n_cols = input_array.shape
    # rows = n_samples, cols = n_features

    # the first column is id (not interested)
    # the second column is Y (label)
    # the other columns are X (features)
    id = input_array[:, 0]
    Y = input_array[:, 1]
    X = input_array[:, 2:]

    return X, Y, id


training_X, training_Y, training_id = process_array(training_data)
testing_X, testing_Y, testing_id = process_array(testing_data)

# check the 1s are added at the last column
# print(testing_X[:, -1])


# since there are only 2 class labels, we only need to find P(Y=1)
# P(Y=0) = 1 - P(Y=1) here, also P(Y=1) used in learning variables
def find_P_Y1(X: np.array, W: np.array) -> np.array:
    # This function gives the logistic regression results based of given data (feature values) and parameters.
    # Inputs:
    # X:     a numpy array of float that containing all input feature values with last column of ones added.
    # W:     a numpy array containing the parameters for logistic regression.
    # Outputs:
    # p_y_1: a numpy array of logistic regression results, giving probability of each sample labeled as 1.
    if X.shape[1] != W.shape[0]:
        print('error in X or W shape')
        return 0

    linear_component = np.dot(X, W)
    probability_y1 = np.exp(linear_component)
    p_y1 = np.divide(probability_y1, probability_y1 + np.ones(probability_y1.shape))

    return p_y1


'''testX = np.array([[1, 2], [3, 4]])
testW = np.array([5, 6])
print(find_P_Y1(testX, testW).shape)'''


def learn_W_MLE(X: np.array, Y: np.array, W_prior: np.array, step_size: float,
                iteration=100, threshold=0.01, interval=100, descent_rate=0.9) -> np.array:
    # This function learns the parameters for logistic regression based on given input data and true labels.
    # Inputs:
    # X:            a numpy array of float that containing all input feature values with last column of ones added.
    # Y:            a numpy array of true labels of the training data (0 and 1 for benign and malignant).
    # step_size:    a float value to set how quickly the parameters were updated by the gradients.
    # iteration:    an integer that gives the maximum number of times the parameters were updated.
    # threshold:    a float value of which if the sums of gradient fells below, the model is believed to have stopped
    #                training and will exit and return the parameters.
    # interval:     an integer set the number of iterations before changing the step size to refine parameters.
    # descent_rate: a float to reduce the step size after interval number of iterations.
    # Outputs:
    # W:            a numpy array containing the parameters for logistic regression after training on given data.

    if X.shape[0] != Y.shape[0]:
        print('error in X or Y shape')
        return 0
    if X.shape[1] != W_prior.shape[0]:
        print('error in X or W prior shape')
        return 0

    # do gradient ascent
    W = W_prior
    converged = False
    for a in range(iteration):
        # print(a)
        # reduce learning rate every interval steps
        if a % interval == interval-1:
            step_size *= descent_rate

        p_y1 = find_P_Y1(X, W)
        y_variation = np.subtract(Y, p_y1)
        gradient = np.dot(X.T, y_variation)
        W = np.add(W, step_size*gradient)

        # stop if the gradient is too close to 0
        update_rate = np.sum(np.absolute(gradient))
        if update_rate < threshold:
            converged = True
            print('steps took to converge: ' + str(a))
            break
    if not converged:
        print('all ' + str(iteration) + ' steps used, not converged by ' + str(threshold))
    return W

'''
testY = np.array([0, 1])
learnedW = learn_W_MLE(testX, testY, testW, 0.1, 10000, 0.0001)
print(learnedW)
print(find_P_Y1(testX, learnedW))
'''


def turn_p_to_y(predicted_results: np.array) -> np.array:
    # This function transformed the probabilities results from the logistic regression into
    # prediction results of 1 (malignant) and o (benign) with cutoff value of 0.5.
    # Inputs:
    # predicted_results: a numpy array containing the predicted probabilities from the logistic regression.
    # Outputs:
    # Y:                 a numpy array containing 0 and 1 for benign and malignant prediction results.
    Y = predicted_results
    size = Y.shape[0]
    for a in range(size):
        if Y[a] >= 0.5:
            Y[a] = 1
        else:
            Y[a] = 0
    return Y


def find_matrix(true_label: np.array, predicted_label: np.array):
    # This function compares the true labels and predicted labels to find true / false positive / negative.
    # Inputs:
    # true_label:          a numpy array containing numbers 0 (benign) and 1 (malignant) for true labels.
    # predicted_label:     a numpy array containing numbers 0 (benign) and 1 (malignant) predicted by the model.
    # Outputs:
    # true_positive:       an integer giving the number of true positives.
    # true_negative:       an integer giving the number of true negatives.
    # false_positive:      an integer giving the number of false positives.
    # false_negative:      an integer giving the number of false negatives.
    # false_positive_list: a list containing indices of false positive predictions.
    # false_negative_list: a list containing indices of false negative predictions.

    if true_label.shape != predicted_label.shape:
        print('error in input shapes')
        return 0
    size = true_label.shape[0]
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    false_positive_list, false_negative_list = [], []
    for a in range(size):
        if predicted_label[a] == 1:
            if true_label[a] == 1:
                true_positive += 1
            else:
                false_positive += 1
                false_positive_list.append(a)
        else:
            if true_label[a] == 0:
                true_negative += 1
            else:
                false_negative += 1
                false_negative_list.append(a)
    return true_positive, true_negative, false_positive, false_negative, false_positive_list, false_negative_list


def add_last_col(input_array: np.array) -> np.array:
    # This function add a column of ones to the end of the input matrix.
    # It makes adding the scalar term easier (expand the parameter vector by 1 for w_0).
    # Input:
    # input_array:  a numpy array matrix of all training data.
    # output_array: a numpy array matrix of the same data with a last column of ones added.

    last_col = np.ones(input_array.shape[0])
    output_array = np.column_stack((input_array, last_col.T))
    return output_array


# add a column of 1 at the end of X for w0
training_X_no_PCA = add_last_col(training_X)
testing_X_no_PCA = add_last_col(testing_X)
#print(testing_X_no_PCA)

# https://builtin.com/data-science/numpy-random-seed
# used to fix random seed
rng = np.random.default_rng(seed=3)
initial_W = rng.random(training_X_no_PCA.shape[1])

learned_W = learn_W_MLE(training_X_no_PCA, training_Y, initial_W, 0.1, 10000, 0.00001)
#print(learned_W)
predicted_Y = find_P_Y1(testing_X_no_PCA, learned_W)
#print(turn_p_to_y(predicted_Y))
#print(testing_Y)

print(find_matrix(testing_Y, turn_p_to_y(predicted_Y)))


num_PCs_needed, variance_covered, principle_components = find_PCs(training_X)
print(num_PCs_needed)
print(variance_covered)
#print(principle_components)
transformed_training_X = PCA_transform(training_X, principle_components)
#print(training_X.shape)
#print(transformed_training_X.shape)
training_X_PCA = add_last_col(transformed_training_X)

learned_W_PCA = learn_W_MLE(training_X_PCA, training_Y, initial_W[:num_PCs_needed + 1], 0.1, 10000, 0.00001)
transformed_testing_X = PCA_transform(testing_X, principle_components)

testing_X_PCA = add_last_col(transformed_testing_X)
#print(transformed_testing_X.shape)
predicted_Y_PCA = find_P_Y1(testing_X_PCA, learned_W_PCA)
print(find_matrix(testing_Y, turn_p_to_y(predicted_Y_PCA)))



