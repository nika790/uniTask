import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def run_edge_detection(image_path):
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image from '{image_path}'. Make sure the file exists in your project directory.")
        return

    # Blur slightly to reduce high-frequency noise
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)

    # --- 2. Apply Edge Detection Methods ---

    # A. Prewitt Filter (using SciPy convolution)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = convolve(blurred_img.astype(float), kernel_x)
    prewitt_y = convolve(blurred_img.astype(float), kernel_y)
    prewitt_edges = np.hypot(prewitt_x, prewitt_y)

    # B. Sobel Filter (OpenCV)
    sobel_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)

    # C. Scharr Filter (OpenCV)
    scharr_x = cv2.Scharr(blurred_img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(blurred_img, cv2.CV_64F, 0, 1)
    scharr_edges = cv2.magnitude(scharr_x, scharr_y)

    # D. Canny Edge Detector (OpenCV)
    canny_edges = cv2.Canny(blurred_img, 100, 200)

    # E. Laplacian Filter (OpenCV)
    laplacian_edges = cv2.Laplacian(blurred_img, cv2.CV_64F)
    laplacian_edges = np.abs(laplacian_edges)

    # --- 3. Visualize Results ---
    titles = [
        'Original Image', 'Prewitt Filter',
        'Sobel Filter', 'Scharr Filter',
        'Canny Detector', 'Laplacian Filter'
    ]
    images = [img, prewitt_edges, sobel_edges, scharr_edges, canny_edges, laplacian_edges]

    plt.figure(figsize=(14, 8))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_edge_detection(r'C:\Users\Home\Downloads\edgedet.png')