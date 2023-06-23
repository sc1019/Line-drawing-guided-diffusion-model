import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def load_and_process_images(input_folder, output_folder):
    # Create a transform for resizing and tensor conversion
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Check if CUDA is available and if yes, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loop over all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load image
            img = Image.open(os.path.join(input_folder, filename))

            # Apply transform (resize + convert to tensor)
            img_tensor = transform(img)

            # Make sure the image tensor has 3 channels
            if img_tensor.shape[0] != 3:
                img_tensor = img_tensor.repeat(3, 1, 1)

            # Expand dimensions to represent a batch of size 1
            img_tensor = img_tensor.unsqueeze(0)

            # Move tensor to the appropriate device
            img_tensor = img_tensor.to(device)

            # Ensure the tensor is a float
            og_img = img_tensor.float()

            # Normalize tensor to 0-255 and convert to CPU
            og_img = ((og_img + 1) * 127.5).byte().cpu()

            # Convert tensor to numpy array and to grayscale
            numpy_img = og_img.squeeze().numpy().transpose(1, 2, 0)
            gray_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)

            # Reduce noise
            blur = cv2.GaussianBlur(gray_img, (5,5), 0)

            # Apply threshold to make image black and white (Otsu's method)
            high_thresh, img_bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Calculate the lower threshold for Canny
            low_thresh = 0.5 * high_thresh

            # Use Canny edge detection to highlight edges
            edges = cv2.Canny(img_bw, low_thresh, high_thresh, apertureSize = 5, L2gradient = True)
            
            # Save edge image to output folder (assuming output_folder exists)
            output_path = os.path.join(output_folder, f'edge_{filename}')
            cv2.imwrite(output_path, edges)

input_folder = 'test_img'
output_folder = 'edge_img'
load_and_process_images(input_folder, output_folder)
