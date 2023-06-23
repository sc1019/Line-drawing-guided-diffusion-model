import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import time
import torchvision.transforms.functional as TF
import torch.linalg
from usage import *
from torch.utils import data
from torchvision import transforms, utils
from torchvision import datasets
from tqdm import tqdm
from einops import rearrange
import torchgeometry as tgm
import glob
import os
from torch import linalg as LA
from scipy.ndimage import zoom as scizoom
from kornia.color.gray import rgb_to_grayscale
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


class ForwardProcessBase:
    
    def forward(self, x, i):
        pass

    @torch.no_grad()
    def reset_parameters(self, batch_size=32):
        pass


class Impasto(ForwardProcessBase):

    def __init__(self, 
                 min_number_of_color=20,
                 max_number_of_color=256,
                 num_timesteps=50,
                 channels=3):
        self.min_number_of_color = min_number_of_color
        self.max_number_of_color = max_number_of_color
        self.num_timesteps = num_timesteps
        self.channels = channels
        self.device_of_kernel = 'cuda'


    def forward(self,input_tensor, step,og=None):
        n_colors = self.max_number_of_color-step+self.min_number_of_color
        # Ensure tensor is on CPU for numpy compatibility
        input_tensor = input_tensor.cpu()

        batch_size, _, height, width = input_tensor.shape

        # Initialize tensor to store quantized images
        quantized_tensor = torch.empty_like(input_tensor)

        # Loop through the batch
        for i in range(batch_size):
            # Extract image from batch and rescale from [-1, 1] to [0, 1]
            image_tensor = (input_tensor[i] + 1) / 2

            # Reshape from (channels, height, width) to (height, width, channels) and scale to 0-255
            image_tensor = image_tensor.permute(1, 2, 0).mul(255).byte()

            # Convert to PIL image
            image_pil = Image.fromarray(image_tensor.numpy(), 'RGB')

            # Quantize image
            quantized_image_pil = image_pil.quantize(colors=n_colors)

            # Convert back to tensor and rescale to [-1, 1]
            quantized_image_tensor = torch.from_numpy(np.array(quantized_image_pil.convert('RGB'))).float().div(255).permute(2, 0, 1) * 2 - 1

            # Store in the batch tensor
            quantized_tensor[i] = quantized_image_tensor

        # utils.save_image(quantized_tensor, f'sample-color-{n_colors}.png', nrow=6)
        return quantized_tensor.cuda()

    # def forward(self,input_tensor, step,og = None):
    #     n_colors = 255
    #     # Ensure tensor is on CPU for numpy compatibility
    #     input_tensor = input_tensor.cpu()

    #     # Get the original shape of the tensor
    #     original_shape = input_tensor.shape

    #     # Rescale input tensor values from [-1, 1] to [0, 1]
    #     rescaled_input = (input_tensor + 1) / 2
    #     # Reshape the tensor to -1,3 (which is equivalent to a list of RGB values)
    #     pixels = rescaled_input.view(-1, 3)

    #     # Convert tensor to numpy for sklearn compatibility
    #     pixels_np = pixels.numpy()

    #     # Perform k-means clustering to find the most dominant colors
    #     kmeans = KMeans(n_clusters=n_colors, n_init=10)
    #     kmeans.fit(pixels_np)
        
    #     # Replace each pixel with its nearest color from the k-means clustering
    #     new_pixels_np = kmeans.cluster_centers_[kmeans.labels_]

    #     # Convert back to tensor
    #     new_pixels = torch.from_numpy(new_pixels_np)

    #     # Reshape the pixels to the original tensor shape
    #     new_tensor = new_pixels.view(original_shape)

    #     # Rescale values from [0, 1] to [-1, 1]
    #     utils.save_image(new_tensor, f'sample-color-{n_colors}.png', nrow=6)
    #     new_tensor = new_tensor * 2 - 1
    #     return new_tensor


class DeColorization(ForwardProcessBase):

    def __init__(self, 
                 decolor_routine='Constant', 
                 decolor_ema_factor=0.9,
                 decolor_total_remove=False,
                 num_timesteps=50,
                 channels=3,
                 to_lab=False):

        self.decolor_routine = decolor_routine
        self.decolor_ema_factor = decolor_ema_factor
        self.decolor_total_remove = decolor_total_remove
        self.channels = channels
        self.num_timesteps = num_timesteps
        self.device_of_kernel = 'cuda'
        self.kernels = self.get_kernels()
        self.to_lab = to_lab

    def get_conv(self, decolor_ema_factor):
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0, padding_mode='circular',
                         bias=False)
        with torch.no_grad():
            ori_color_weight = torch.eye(self.channels)[:, :, None, None]
            decolor_weight = torch.ones((self.channels, self.channels)) / float(self.channels)
            decolor_weight = decolor_weight[:, :, None, None]
            kernel = decolor_ema_factor * ori_color_weight + (1.0 - decolor_ema_factor) * decolor_weight
            conv.weight = nn.Parameter(kernel)

        if self.device_of_kernel == 'cuda':
            conv = conv.cuda()

        return conv

    def get_kernels(self):
        kernels = []

        if self.decolor_routine == 'Constant':
            for i in range(self.num_timesteps):
                if i == self.num_timesteps - 1 and self.decolor_total_remove:
                    kernels.append(self.get_conv(0.0)) 
                else:
                    kernels.append(self.get_conv(self.decolor_ema_factor))
        elif self.decolor_routine == 'Linear':
            diff = 1.0 / self.num_timesteps
            start = 1.0
            for i in range(self.num_timesteps):
                if i == self.num_timesteps - 1 and self.decolor_total_remove:
                    kernels.append(self.get_conv(0.0)) 
                else:
                    # start * (1 - ema_factor) = diff
                    # ema_factor = 1 - diff / start
                    ema_factor = 1 - diff / start
                    start = start * ema_factor
                    kernels.append(self.get_conv(ema_factor))
        return kernels

    def forward(self, x, i, og=None):
        if self.to_lab:
            x_rgb = lab2rgb(x)
            x_next = self.kernels[i](x_rgb)
            return rgb2lab(x_next)
        else:
            img = self.kernels[i](x)
            return img
    
    def total_forward(self, x_in):
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0, padding_mode='circular',
                         bias=False)
        if self.to_lab:
            x = lab2rgb(x_in)
        else:
            x = x_in

        with torch.no_grad():
            decolor_weight = torch.ones((self.channels, self.channels)) / float(self.channels)
            decolor_weight = decolor_weight[:, :, None, None]
            kernel = decolor_weight
            conv.weight = nn.Parameter(kernel)

        if self.device_of_kernel == 'cuda':
            conv = conv.cuda()

        x_out = conv(x)
        if self.to_lab:
            x_out = rgb2lab(x_out)
        return x_out


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
class EdgeDetection(ForwardProcessBase):

    def __init__(self, 
                 min_number_of_color=20,
                 max_number_of_color=256,
                 num_timesteps=50,
                 channels=3,
                 canny_threshold1=100,
                 canny_threshold2=200):
        self.min_number_of_color = min_number_of_color
        self.max_number_of_color = max_number_of_color
        self.num_timesteps = num_timesteps
        self.channels = channels
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2

    def forward(self,x, step, og_img=None):
        # Ensure the tensor is a float
        og_img = og_img.float()

        # Normalize tensor to 0-255 and convert to CPU
        og_img = ((og_img + 1) * 127.5).byte().cpu()

        # Initialize a list to hold the output
        output_list = []

        # Loop over the batch
        for i in range(og_img.shape[0]):

            # Convert tensor to numpy array and to grayscale
            numpy_img = og_img[i].numpy().transpose(1, 2, 0)
            gray_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)

            # Reduce noise
            blur = cv2.GaussianBlur(gray_img, (5,5), 0)

            # Apply threshold to make image black and white (Otsu's method)
            high_thresh, img_bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Calculate the lower threshold for Canny
            low_thresh = 0.5 * high_thresh

            # Use Canny edge detection to highlight edges
            edges = cv2.Canny(img_bw, low_thresh, high_thresh, apertureSize = 5, L2gradient = True)
            
            # Create a sharpening filter
            sharpen_kernel = np.array([[-1,-1,-1], [-1,15,-1], [-1,-1,-1]])

            # Apply the sharpening filter to the edges
            sharpened = cv2.filter2D(edges, -1, sharpen_kernel)

            # Rescale the sharpened image to [0, 1] range
            sharpened_normalized = sharpened / 255.0

            # Convert the sharpened image back to PyTorch tensor
            sharpened_tensor = torch.from_numpy(sharpened_normalized).float()
            sharpened_tensor = 1 - sharpened_tensor
            # Blend the original image with the edge-detected image
            weight = (step+1) / self.num_timesteps
            blended_image = og_img[i].float() / 255.0 * (1 - weight) + sharpened_tensor.unsqueeze(0) * weight

            # Append to the output list
            output_list.append(blended_image)

        # Stack the output list to a tensor
        output_tensor = torch.stack(output_list).cuda()
        return output_tensor
