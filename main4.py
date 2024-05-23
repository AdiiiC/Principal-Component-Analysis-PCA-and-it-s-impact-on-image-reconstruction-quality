import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Load the Olivetti Faces dataset
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
faces_images = faces_data.images
faces_data = faces_data.data

# Flatten the images into vectors
faces_data_flatten = faces_data.reshape((faces_data.shape[0], -1))

# Standardize the data (mean=0, variance=1)
mean_face = np.mean(faces_data_flatten, axis=0)
std_faces_data = (faces_data_flatten - mean_face) / np.std(faces_data_flatten, axis=0)

# Compute the covariance matrix
cov_matrix = np.cov(std_faces_data, rowvar=False)

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvectors by eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sorted_indices]

# Vary num_components
num_components_range = range(1, 100, 5)
mse_values_list = []

# Iterate over different values of num_components
for num_components in num_components_range:
    # Choose the number of principal components (eigenfaces) to use
    selected_eigenvectors = eigenvectors[:, :num_components]

    # Project the standardized faces data onto the selected eigenvectors
    projected_faces_data = np.dot(std_faces_data, selected_eigenvectors)

    # Reconstruct the faces from the projected data
    reconstructed_faces_data = np.dot(projected_faces_data, selected_eigenvectors.T)

    # Rescale the reconstructed faces to the original scale
    reconstructed_faces_data = (reconstructed_faces_data * np.std(faces_data_flatten, axis=0)) + mean_face

    # Calculate the Mean Squared Error (MSE) for each face
    mse_values = np.mean((faces_data_flatten - reconstructed_faces_data)**2, axis=1)

    # Append MSE values for the current num_components to the list
    mse_values_list.append(np.mean(mse_values))

# Plot the graph
plt.plot(num_components_range, mse_values_list, marker='o')
plt.title('num_components vs Mean Squared Error (MSE)')
plt.xlabel('num_components')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.show()