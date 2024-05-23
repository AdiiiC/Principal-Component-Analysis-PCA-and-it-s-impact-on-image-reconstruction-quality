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

# Choose the number of principal components (eigenfaces) to use
num_components = 40
selected_eigenvectors = eigenvectors[:, :num_components]

# Project the standardized faces data onto the selected eigenvectors
projected_faces_data = np.dot(std_faces_data, selected_eigenvectors)

# Scatter plot of projections onto PC5 and PC6 for images 2 and 7
image_indices = [1, 6]  # Indices of the images to be projected (0-based)

plt.figure(figsize=(8, 6))

for i in image_indices:
    # Project the selected image onto PC5 and PC6
    selected_image_std = (faces_data_flatten[i] - mean_face) / np.std(faces_data_flatten, axis=0)
    projected_image_data = np.dot(selected_image_std, selected_eigenvectors[:, 4:6])

    # Scatter plot
    plt.scatter(projected_image_data[0], projected_image_data[1], label=f"Image {i+1}")

plt.title('Scatter Plot of Projections onto PC5 and PC6')
plt.xlabel('PC5')
plt.ylabel('PC6')
plt.legend()
plt.show()