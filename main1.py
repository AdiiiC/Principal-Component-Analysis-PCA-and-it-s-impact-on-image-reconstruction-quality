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
num_components = 50
selected_eigenvectors = eigenvectors[:, :num_components]

# Project the standardized faces data onto the selected eigenvectors
projected_faces_data = np.dot(std_faces_data, selected_eigenvectors)

# Reconstruct the faces from the projected data
reconstructed_faces_data = np.dot(projected_faces_data, selected_eigenvectors.T)

# Rescale the reconstructed faces to the original scale
reconstructed_faces_data = (reconstructed_faces_data * np.std(faces_data_flatten, axis=0)) + mean_face

# Calculate the PSNR for each face
psnr_values = []
for i in range(faces_data_flatten.shape[0]):
    original_face = faces_data_flatten[i].reshape(faces_images[0].shape)
    reconstructed_face = reconstructed_faces_data[i].reshape(faces_images[0].shape)

    # Calculate the mean squared error
    mse = np.mean((original_face - reconstructed_face)**2)

    # Calculate PSNR
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    psnr_values.append(psnr)

# Print the PSNR values
print("PSNR Values:")
for i, psnr in enumerate(psnr_values):
    print(f"Face {i+1}: {psnr:.2f} dB")

# Plot the original and reconstructed faces
n_faces = 5
fig, axes = plt.subplots(n_faces, 2, figsize=(5, 10),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i in range(n_faces):
    axes[i, 0].imshow(faces_data_flatten[i].reshape(faces_images[0].shape), cmap='gray')
    axes[i, 1].imshow(reconstructed_faces_data[i].reshape(faces_images[0].shape), cmap='gray')

axes[0, 0].set_title('Original Faces')
axes[0, 1].set_title('Reconstructed Faces')
plt.show()