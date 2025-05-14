import os
import cv2
import numpy as np
# Directories
input_dir  =  r"E:\APPS\PythonDataSets\caltech\caltech-101\101_ObjectCategories\101_ObjectCategories"
output_dir = r"E:\APPS\PythonDataSets\caltech\caltech-101\101_ObjectCategories\filtered_preprocessed_categories"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Parameters for filters
gaussian_kernel_size = (13, 13)  # Adjust kernel size for Gaussian filter
laplacian_ksize = 3  # Kernel size for Laplacian filter
A = 1.5  # Scaling constant for sharpening

# Loop through all images in the dataset
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)
    
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Apply Gaussian filter
                img_gaussian = cv2.GaussianBlur(img, gaussian_kernel_size, 0)
                
                # Apply Laplacian filter
                img_laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F, ksize=laplacian_ksize)
                img_laplacian = cv2.convertScaleAbs(img_laplacian)  # Convert to 8-bit image
                
                # Add Laplacian to the original image with constant A
                img_sharpened = cv2.addWeighted(img, 1, img_laplacian, A, 0)
                
                # Save the processed image
                output_path = os.path.join(output_class_path, os.path.splitext(img_name)[0] + ".jpg")
                try:
                    cv2.imwrite(output_path, img_sharpened)
                except Exception as e:
                    print(f"Error saving image {output_path}: {e}")
                    continue

print("Processing completed. Filtered images are saved in:", output_dir)
