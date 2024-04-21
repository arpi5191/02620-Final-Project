import numpy as np


def find_PCs(centered_data: np.array, variance_cover=0.9) -> (int, float, np.array):
    N = centered_data.shape[0]
    cov_matrix = 1/N * np.dot(centered_data.T, centered_data)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    total_var = np.sum(eigenvalues)

    PCs_needed = 1
    PCs = eigenvectors[-1]
    var_covered = eigenvalues[-1] / total_var
    # while var_covered < variance_cover:
    for i in range(7):
        PCs_needed += 1
        PCs = np.row_stack((PCs, eigenvectors[-PCs_needed]))
        var_covered += eigenvalues[-PCs_needed] / total_var

    return PCs_needed, var_covered, PCs


def PCA_transform(centered_data: np.array, PCs: np.array) -> np.array:
    projected_data = np.dot(centered_data, PCs.T)
    return projected_data
