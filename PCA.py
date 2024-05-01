# Import packages
import numpy as np

# find_PCs: Obtain the PCs the dataset
def find_PCs(centered_data: np.array, variance_cover=0.9) -> (int, float, np.array):

    # Find the dimensions of the centered data
    N = centered_data.shape[0]

    # Compute the covariance matrix
    cov_matrix = 1/N * np.dot(centered_data.T, centered_data)

    # Perform eigendecomposition to obtain the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Find the sum of the eigenvectors
    total_var = np.sum(eigenvalues)

    # Initialize the following parameters
    PCs_needed = 1
    PCs = eigenvectors[-1]
    var_covered = eigenvalues[-1] / total_var

    # Find the PCs and stack them in the matrix
    while var_covered < variance_cover:
        PCs_needed += 1
        PCs = np.row_stack((PCs, eigenvectors[-PCs_needed]))
        var_covered += eigenvalues[-PCs_needed] / total_var

    # Return the PCs needed, the variance covered and the PCs
    return PCs_needed, var_covered, PCs

# Project the data
def PCA_transform(centered_data: np.array, PCs: np.array) -> np.array:

    # Compute the dot product of the data point and the transpose of the PCs
    projected_data = np.dot(centered_data, PCs.T)

    # Return the projected data
    return projected_data

