#!/usr/bin/env python3
"""
Preprocess Registered Files - Final Production Version
====================================================

This script standardizes the input nifti imaging file spatially to reach a 1*1*N stack of data arrays.
It uses pixel spacing for rescaling to reach 1*1 resolution.
Then, it employs zero-padding and/or cropping to fixate the dimension on 256*256. 

System Requirements:
- Python 3.9+
- 

Features:
- 

Author: Mahdi Bashiri Bawil
Date: September 2025
Version: 1.0 (Production Ready)
"""


import os
import cv2
import shutil
import skimage
import warnings
import numpy as np
import nibabel as nib
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter, binary_dilation, label
from skimage.morphology import diamond, binary_opening, binary_closing, erosion, dilation, rectangle, disk, remove_small_objects

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


def size_check(data_all, v_size, dim=(256, 256)):
    padded_data_all = np.zeros((dim[0], dim[1], data_all.shape[2]))

    for i in range(data_all.shape[2]):

        data_ = data_all[..., i]

        # Assuming 'data_' is the 2D image matrix
        # Here, we define scaling factors for each dimension
        data_shape = data_.shape
        scaling_factors = v_size[:2]

        rescaled_data = rescale(data_, scaling_factors, anti_aliasing=True, mode='reflect')

        # The rescaled_image variable now contains the rescaled image matrix wit isometric (1,1,X) voxels in imaging plane
        image_shape = rescaled_data.shape

        # Define the desired dimensions for padding
        desired_shape = dim  # Specify the specific dimensions you want to reach
        image = np.zeros((dim))

        # Calculate the amount of padding needed for each dimension
        pad_h = desired_shape[0] - image_shape[0]
        if pad_h < 0:
            pad_height = 0
        else:
            pad_height = pad_h

        pad_w = desired_shape[1] - image_shape[1]
        if pad_w < 0:
            pad_width = 0
        else:
            pad_width = pad_w

        # Calculate the padding configuration
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the image symmetrically to reach the desired dimension
        padded_data = np.pad(rescaled_data, ((pad_top, pad_bottom), (pad_left, pad_right)),
                             mode='constant', constant_values=np.min(rescaled_data))

        # Truncate the padded image to fit into desired dim size
        if pad_h < 0:
            image = padded_data[int(-pad_height / 2):desired_shape[0] + int(-pad_height / 2), :]
            padded_data = image
        if pad_w < 0:
            image = padded_data[:, int(-pad_width / 2):desired_shape[1] + int(-pad_width / 2)]
            padded_data = image

        padded_data_all[..., i] = padded_data

    return padded_data_all


def load_nifti(file_path):
    """Load a NIfTI file and return the image data and the nibabel object."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img


def save_nifti(data, ref_img, out_path):
    """Save data as a NIfTI file using a reference image for header/affine."""
    new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, out_path)
    print(f"Saved pre-processed data to {out_path}")


def preprocess(
    data_dir="/mnt/e/MBashiri/Thesis/p6/Data/MS_100_patient_registered",
    save_dir="/mnt/e/MBashiri/Thesis/p6/Data/MS_100_patient_preprocessed",
    desired_dim=256
    ):

    os.makedirs(save_dir, exist_ok=True)

    folders = os.listdir(data_dir)
    for folder in folders:

        folder_dir = os.path.join(data_dir, folder)

        output_dir = os.path.join(save_dir, folder)
        os.makedirs(output_dir, exist_ok=True)
            
        flair_file = [f for f in os.listdir(folder_dir) if f.endswith('_FLAIR.nii.gz')]
        flair_file = flair_file[0]

        # Handling all three input files per patient
        flair_file_path = os.path.join(folder_dir, flair_file)
        t1_file_path = os.path.join(folder_dir, flair_file.replace('_FLAIR.nii.gz', '_T1WI_reg.nii.gz'))
        t2_file_path = os.path.join(folder_dir, flair_file.replace('_FLAIR.nii.gz', '_T2WI_reg.nii.gz'))

        file_set = [
            flair_file_path,
            t1_file_path,
            t2_file_path
        ]

        first_time = True       # Controls the voxel size extracted only from the FLAIR file and applicable for other two files. We do that for procedure consistency.

        for file in file_set:

            #
            # ## Loading data

            nifti_data, nifti_obj = load_nifti(file)

            #
            # ##  Pre-processing
            if first_time:
                voxel_size = nifti_obj.header.get_zooms()[:3]  # (x, y, z) in mm
                print(voxel_size, '\n')
                first_time = False

            # Forced
            voxel_size = np.array(voxel_size)
            voxel_size[:-1] = 1
            nifti_data_s = size_check(nifti_data, voxel_size, dim=(desired_dim, desired_dim))

            # Save the spatially-normalized data, the iso file named after isometric size checking. You might want to copy the header/affine from the original data.
            iso_file_path = os.path.join(output_dir, os.path.basename(file).replace('.nii.gz', '.nii.gz'))
            save_nifti(nifti_data_s, nifti_obj, iso_file_path)

            # Save slice samples
            sample_list = [4, 9]
            sample_list = np.arange(0, np.shape(nifti_data_s)[-1])
            for idx in sample_list:
                img = nifti_data_s[..., idx]
                img = (img - np.max(img)) / (np.max(img) - np.min(img))
                img = np.uint8(255.0 * img)
                skimage.io.imsave(iso_file_path.replace('.nii.gz', f'_{idx+1}.png'), img)

if __name__ == '__main__':

    preprocess(
        data_dir="/mnt/e/MBashiri/Thesis/p6/Data/MS_100_patient_registered",
        save_dir="/mnt/e/MBashiri/Thesis/p6/Data/MS_100_patient_preprocessed",
        desired_dim=256
        )
