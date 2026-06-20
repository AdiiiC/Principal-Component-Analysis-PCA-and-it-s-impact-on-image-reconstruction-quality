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

# Reconstruct the faces from the projected data
reconstructed_faces_data = np.dot(projected_faces_data, selected_eigenvectors.T)

# Rescale the reconstructed faces to the original scale
reconstructed_faces_data = (reconstructed_faces_data * np.std(faces_data_flatten, axis=0)) + mean_face

# Plot the original and reconstructed faces for images 2 and 7 using PC5 and PC6
image_indices = [1, 6]  # Indices of the images to be projected (0-based)

fig, axes = plt.subplots(len(image_indices), 2, figsize=(5, 5),
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for idx, i in enumerate(image_indices):
    # Project the selected image onto PC5 and PC6
    selected_image_std = (faces_data_flatten[i] - mean_face) / np.std(faces_data_flatten, axis=0)
    projected_image_data = np.dot(selected_image_std, selected_eigenvectors[:, 4:6])

    # Reconstruct the image from the projected data
    reconstructed_image_data = np.dot(projected_image_data, selected_eigenvectors[:, 4:6].T)

    # Rescale the reconstructed image to the original scale
    reconstructed_image_data = (reconstructed_image_data * np.std(faces_data_flatten, axis=0)) + mean_face

    axes[idx, 0].imshow(faces_data_flatten[i].reshape(faces_images[0].shape), cmap='gray')
    axes[idx, 1].imshow(reconstructed_image_data.reshape(faces_images[0].shape), cmap='gray')

axes[0, 0].set_title('Original Faces')
axes[0, 1].set_title('Reconstructed Faces (PC5 and PC6)')
plt.show()
