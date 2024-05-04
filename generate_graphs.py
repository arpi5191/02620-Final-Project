import matplotlib.pyplot as plt
import numpy as np
from PCA import find_PCs, PCA_transform
from logistic_regression import *


def separate_space(space_array: np.array, label_array: np.array, false_pos: list, false_neg: list) -> np.array:
    # This function returns four lists of coordinates to help drawing the scatter plot.
    # Inputs:
    # space_array: a numpy array that have all data projected on the top two PCs.
    # label_array: a numpy array with the true labels as 1 and 0.
    # false_pos:   a numpy array with the indices of false positive predictions.
    # false_neg:   a numpy array with the indices of false negative predictions.
    # Outputs:
    # Four numpy arrays that contains the x and y coordinates for true positive, true negative,
    # false positive, and false positive prediction results.
    if space_array.shape[1] != label_array.shape[0]:
        print('error in input spaces')
        return 0

    separated_M, separated_B = np.zeros((space_array.shape[0], 1)), np.zeros((space_array.shape[0], 1))
    false_negative_data, false_positive_data = np.zeros((space_array.shape[0], 1)), np.zeros((space_array.shape[0], 1))
    size = space_array.shape[1]
    for a in range(size):
        if label_array[a] == 1:
            if a not in false_neg:
                separated_M = np.column_stack((separated_M, space_array[:, a]))
            else:
                false_negative_data = np.column_stack((false_negative_data, space_array[:, a]))
        else:
            if a not in false_pos:
                separated_B = np.column_stack((separated_B, space_array[:, a]))
            else:
                false_positive_data = np.column_stack((false_positive_data, space_array[:, a]))

    return separated_M, separated_B, false_positive_data, false_negative_data


def draw_scatter_plot(set_M, set_B, n_true_p, n_true_n, n_false_p, n_false_n, set_FP, set_FN, x1, x2, c, title: str):
    # This function takes in the sets of coordinates and other values to draw the scatter plot and decision boundary.
    # Inputs:
    # set_M:     a numpy array with coordinates for true positive predictions.
    # set_B:     a numpy array with coordinates for true negative predictions.
    # n_true_p:  an integer with the number of true positive predictions.
    # n_true_n:  an integer with the number of true negative predictions.
    # n_false_p: an integer with the number of false positive predictions.
    # n_false_n: an integer with the number of false negative predictions.
    # set_FP:    a list with coordinates for false positive predictions.
    # set_FN:    a list with coordinates for false negative predictions.
    # equation for decision boundary x_1 * x + x_2 * y + c = 0
    # x_1:       a float number giving the x_1 value in the decision boundary equation.
    # x_2:       a float number giving the x_2 value in the decision boundary equation.
    # c:         a float number giving the c value in the decision boundary equation.
    # title:     a string that set the title of the generated graph.
    # Outputs:
    # Shows the scatter graph generated.
    # Save the graph in the same folder with name given.

    plt.axline((0, -c / x2), slope=(x1 / (-x2)))

    x_d = set_M[0]
    y_d = set_M[1]
    plt.scatter(x_d, y_d, color='red', label="true positive " + str(n_true_p))

    x_d = set_B[0]
    y_d = set_B[1]
    plt.scatter(x_d, y_d, color='blue', label="true negative " + str(n_true_n))

    x_d = set_FN[0]
    y_d = set_FN[1]
    plt.scatter(x_d, y_d, color='lightblue', label="false negative " + str(n_false_n))

    x_d = set_FP[0]
    y_d = set_FP[1]
    plt.scatter(x_d, y_d, color='orange', label="false positive " + str(n_false_p))

    plt.legend()

    plt.title(title)
    plt.xlabel("First PC")
    plt.ylabel("Second PC")

    plt.savefig("Results/" + "LogReg/" + title + ".png")
    # plt.show()
    plt.close()


top_2_PCs_space = transformed_testing_X[:, :2].T

true_p, true_n, false_p, false_n, false_p_list, false_n_list = \
    find_matrix(testing_Y, turn_p_to_y(predicted_Y))

true_p_PCA, true_n_PCA, false_p_PCA, false_n_PCA, false_p_PCA_list, false_n_PCA_list = \
    find_matrix(testing_Y, turn_p_to_y(predicted_Y_PCA))

w_space = learned_W[:-1]
# determine how the top 2 PCs directions contribute in the non_PCA model
projected_w = PCA_transform(w_space, principle_components)
# use that to graph the decision boundary
ORI_x = projected_w[0]
ORI_y = projected_w[1]
ORI_c = learned_W[-1]
s_M, s_B, s_FP, s_FN = separate_space(top_2_PCs_space, testing_Y, false_p_list, false_n_list)
draw_scatter_plot(s_M, s_B, true_p, true_n, false_p, false_n, s_FP,
                  s_FN, ORI_x, ORI_y, ORI_c, 'Logistic Regression No PC Transform')


PCA_x = learned_W_PCA[0]
PCA_y = learned_W_PCA[1]
PCA_c = learned_W_PCA[learned_W_PCA.shape[0] - 1]
s_M_PCA, s_B_PCA, s_FP_PCA, s_FN_PCA = separate_space(top_2_PCs_space, testing_Y, false_p_PCA_list, false_n_PCA_list)

draw_scatter_plot(s_M_PCA, s_B_PCA, true_p_PCA, true_n_PCA, false_p_PCA, false_n_PCA,
                  s_FP_PCA, s_FN_PCA, PCA_x, PCA_y, PCA_c, 'Logistic Regression PC Transformed')

# This is the parameter vector returned by the SVM model.
SVM_no_PCA_parameter = np.array([0.32677228, 0.23701241, 0.32718588, 0.35936443, -0.03647987, -0.27299968,
                                 0.54405546, 0.57877149, 0.08416435, -0.21571561,  0.47370876, -0.19401536,
                                 0.41160339, 0.4138777, 0.1664169, -0.36243802, -0.07315217, 0.16143742,
                                 -0.10545119, -0.2337695, 0.34912484, 0.63110539, 0.33429821, 0.38342793,
                                 0.39324234, -0.12576957, 0.40234214, 0.31383452, 0.45889848, 0.31271299])
projected_SVM_w_no_PCA = PCA_transform(SVM_no_PCA_parameter, principle_components)
SVM_ORI_x = projected_SVM_w_no_PCA[0]
SVM_ORI_y = projected_SVM_w_no_PCA[1]
SVM_ORI_c = -0.1871037930155531 * SVM_ORI_y
# This is the prediction results from the SVM model.
SVM_no_PCA_res = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
                           0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                           0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
                           0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                           1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                           1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0])
true_p_SVM_no_PCA, true_n_SVM_no_PCA, false_p_SVM_no_PCA, false_n_SVM_no_PCA, \
    false_p_SVM_no_PCA_list, false_n_SVM_no_PCA_list = find_matrix(testing_Y, SVM_no_PCA_res)
s_M_SVM_no_PCA, s_B_SVM_no_PCA, s_FP_SVM_no_PCA, s_FN_SVM_no_PCA \
    = separate_space(top_2_PCs_space, testing_Y, false_p_SVM_no_PCA_list, false_n_SVM_no_PCA_list)

draw_scatter_plot(s_M_SVM_no_PCA, s_B_SVM_no_PCA, true_p_SVM_no_PCA, true_n_SVM_no_PCA,
                  false_p_SVM_no_PCA, false_n_SVM_no_PCA, s_FP_SVM_no_PCA, s_FN_SVM_no_PCA,
                  SVM_ORI_x, SVM_ORI_y, SVM_ORI_c, 'SVM no PC Transform')


PCA_SVM_x = -0.14823298366348833
PCA_SVM_y = -0.12658642673890808
PCA_SVM_c = -0.43799744259361645 * PCA_SVM_y


SVM_res = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
                    0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                    1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 0])
true_p_SVM, true_n_SVM, false_p_SVM, false_n_SVM, false_p_SVM_list, false_n_SVM_list = \
    find_matrix(testing_Y, SVM_res)
s_M_SVM_PCA, s_B_SVM_PCA, s_FP_SVM_PCA, s_FN_SVM_PCA \
    = separate_space(top_2_PCs_space, testing_Y, false_p_SVM_list, false_n_SVM_list)

draw_scatter_plot(s_M_SVM_PCA, s_B_SVM_PCA, true_p_SVM, true_n_SVM, false_p_SVM, false_n_SVM,
                  s_FP_SVM_PCA, s_FN_SVM_PCA, PCA_SVM_x, PCA_SVM_y, PCA_SVM_c, 'SVM PC Transformed')

