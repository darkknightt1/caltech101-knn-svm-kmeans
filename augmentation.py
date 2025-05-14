
import os
import random
import cv2
import numpy as np
from PIL import Image
import os

def rename_images(directory, start_index=436):
    """
    Rename all image files in a directory with sequential names starting from a given index.
    
    Args:
        directory (str): Path to the directory containing image files.
        start_index (int): Starting index for naming the images.
    """
    # Get a sorted list of files in the directory
    files = sorted(os.listdir(directory))
    
    current_index = start_index
    
    for file in files:
        file_path = os.path.join(directory, file)
        
        # Check if the current file is an image (by extension)
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            # Generate the new filename
            new_name = f"image_{current_index}.jpg"
            new_file_path = os.path.join(directory, new_name)
            
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed: {file} -> {new_name}")
            
            # Increment the index
            current_index += 1




def inpaint_black_borders(image):
    """
    Removes black borders by replacing pixels with value 0 (black)
    using the nearest non-zero pixel value.

    Parameters:
    - image (numpy.ndarray): The input image with black borders.

    Returns:
    - numpy.ndarray: The image with black borders removed.
    """
    # Create a mask where black pixels are marked (value of 0)
    mask = cv2.inRange(image, (0, 0, 0), (0, 0, 0))

    # Inpaint the black areas using the nearest pixels
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted_image

def rotate_and_inpaint_images(dataset_path, output_folder=None):
    """
    Augments the dataset by rotating images with random angles and removes black borders.

    Parameters:
    - dataset_path (str): Path to the dataset where each class has its own folder.
    - output_folder (str): If provided, the rotated images will be saved in this folder.
                           Otherwise, images will be saved in the original dataset.

    Returns:
    - None
    """
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        target_folder = output_folder if output_folder else class_folder
        if output_folder:
            os.makedirs(os.path.join(target_folder, class_name), exist_ok=True)

        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                # Open image with OpenCV
                img = cv2.imread(image_path)
                if img is None:
                    continue

                # Rotate the image randomly
                angle = random.uniform(0, 360)
                center = (img.shape[1] // 2, img.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), borderValue=(0, 0, 0))

                # Inpaint black borders
                inpainted_img = inpaint_black_borders(rotated_img)

                # Save the inpainted image
                base_name, ext = os.path.splitext(image_name)
                new_image_name = f"{base_name}_rotated_{int(angle)}{ext}"
                save_path = os.path.join(target_folder, class_name, new_image_name)
                cv2.imwrite(save_path, inpainted_img)

                print(f"Saved rotated and inpainted image: {save_path}")
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")

if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path =  r"E:\APPS\PythonDataSets\caltech\caltech-101\101_ObjectCategories\datasett"
    
    # Optional: Set an output folder if you want to save rotated images separately
    output_folder = None  # Save in the same directory as the original dataset

    rotate_and_inpaint_images(dataset_path, dataset_path)
    
    #directory_path = r"E:\APPS\PythonDataSets\caltech\caltech-101\101_ObjectCategories\dataset"  
    #rename_images(directory_path, start_index=436)
