import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage import data, color

def add_noise(img, noise_type, seed=None):
    """Add noise to the image using random_noise"""
    if seed is not None:
        np.random.seed(seed)  # Set the seed globally
    return random_noise(img, mode=noise_type)

def average_images(image_list):
    """Compute the average of multiple noisy images to reduce noise"""
    return np.mean(image_list, axis=0)

# Load and convert the image to grayscale
coffee = data.coffee()
gray_coffee = color.rgb2gray(coffee)

# Number of noisy samples
num_samples = 10

# Generate noisy images
noisy_gaussian = [add_noise(gray_coffee, 'gaussian', seed=i) for i in range(num_samples)]
averaged_gaussian = average_images(noisy_gaussian)

noisy_saltpepper = [add_noise(gray_coffee, 's&p', seed=i) for i in range(num_samples)]
averaged_saltpepper = average_images(noisy_saltpepper)

# Display results
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes[0, 0].imshow(gray_coffee, cmap='gray')
axes[0, 0].set_title("Original Image")

for i in range(3):
    axes[0, i+1].imshow(noisy_gaussian[i], cmap='gray')
    axes[0, i+1].set_title(f"Gaussian Noise {i+1}")
    
for i in range(3):
    axes[1, i].imshow(noisy_saltpepper[i], cmap='gray')
    axes[1, i].set_title(f"Salt & Pepper Noise {i+1}")

axes[1, 3].imshow(averaged_gaussian, cmap='gray')
axes[1, 3].set_title("Averaged Gaussian Noise")

axes[2, 3].imshow(averaged_saltpepper, cmap='gray')
axes[2, 3].set_title("Averaged Salt & Pepper Noise")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()

# Mathematical explanation in a markdown cell
md_text = """
## Removing Gaussian Noise with Averaging

Assume the original image is represented as \( I \) and the Gaussian noise for the \( i \)-th sample is \( N_i \). Each noisy sample of the image can be expressed as:

\[ I_i = I + N_i \]

If \( N_i \) is Gaussian noise with a mean of zero, averaging over \( M \) samples results in:

\[ \bar{I} = \frac{1}{M} \sum_{i=1}^{M} I_i = \frac{1}{M} \sum_{i=1}^{M} (I + N_i) \]

Since the mean of Gaussian noise is zero, the result converges to the original image:

\[ \bar{I} = I + \frac{1}{M} \sum_{i=1}^{M} N_i \approx I \]

As the number of samples increases, the noise averages out to zero, leaving only the original image.
"""
print(md_text)