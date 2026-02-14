import pandas as pd
import numpy as np
import os
import shutil
import pydicom
import SimpleITK as sitk
import nibabel as nib
from skimage import transform
from natsort import ns, natsorted
import matplotlib.pyplot as plt
import math
import nrrd
import torch
import torch.nn.functional as F
import cv2
from pypinyin import pinyin, lazy_pinyin, Style

'''统一pet/ct影像大小到256*128*128'''
def resize_sitk_3D(image, outputSize=None, interpolator=sitk.sitkLinear):
    inputSize = image.GetSize()
    inputSpacing = image.GetSpacing()
    outputSpacing = [1.0, 1.0, 1.0]
    if outputSize:
        outputSpacing[0] = inputSpacing[0] * (inputSize[0] / outputSize[0])
        outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1])
        outputSpacing[2] = inputSpacing[2] * (inputSize[2] / outputSize[2])
    else:
        outputSize = [0.0, 0.0, 0.0]
        outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
        outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
        outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(outputSize)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    image = resampler.Execute(image)
    return image
def resample_sitkimage(sitk_image_, new_size_):
    org_spacing = sitk_image_.GetSpacing()
    org_size = sitk_image_.GetSize()
    new_space = [(old_size * old_space / new_size) for old_size, old_space, new_size in
                 zip(org_size, org_spacing, new_size_)]
    resampled_img = sitk.Resample(
        sitk_image_,
        new_size_,
        sitk.Transform(),
        sitk.sitkLinear,
        sitk_image_.GetOrigin(),
        new_space,
        sitk_image_.GetDirection(),
        0,
        sitk_image_.GetPixelID()
    )
    return resampled_img

folder_path = '/***/AlignmentData'
out_path = '/***/256*128*128Data'
folder_list = os.listdir(folder_path)
for folder in folder_list:
    print(folder)
    if not os.path.exists(os.path.join(out_path, folder,'CT.nii')) and not os.path.exists(os.path.join(out_path, folder,'PET.nii')):
        try:
            ct_path = os.path.join(folder_path,folder,'NewCT.nrrd')
            pet_path = os.path.join(folder_path,folder,'NewPET.nrrd')
            ct_image = sitk.ReadImage(ct_path)
            pet_image = sitk.ReadImage(pet_path)
            newct_image = resize_sitk_3D(ct_image)
            newpet_image = resize_sitk_3D(pet_image)

            ct_array = sitk.GetArrayFromImage(newct_image)
            pet_array = sitk.GetArrayFromImage(newpet_image)
            final_CT_array = transform.resize(ct_array, (256, 128, 128), order=3, preserve_range=True, anti_aliasing=True)
            final_CT = sitk.GetImageFromArray(final_CT_array)

            final_PET_array = transform.resize(pet_array, (256, 128, 128), order=3, preserve_range=True, anti_aliasing=True)
            final_PET = sitk.GetImageFromArray(final_PET_array)

            if not os.path.exists(os.path.join(out_path, folder)):
                os.makedirs(os.path.join(out_path, folder))
            sitk.WriteImage(final_CT,os.path.join(out_path, folder,'CT.nii'))
            sitk.WriteImage(final_PET, os.path.join(out_path, folder, 'PET.nii'))

        except:
            print(folder,'存在问题')


'''将nii转为npy'''
data_root = '/***'
out_path = '/***'
subforder_list = os.listdir(data_root)
for allnames in subforder_list:
    try:
        print(allnames)
        name1_path = os.path.join(data_root,allnames, 'CT.nii')
        name2_path = os.path.join(data_root,allnames, 'PET.nii')

        img1_nii = sitk.ReadImage(name1_path)
        img2_nii = sitk.ReadImage(name2_path)

        img1_numpy = sitk.GetArrayFromImage(img1_nii)
        img2_numpy = sitk.GetArrayFromImage(img2_nii)

        os.makedirs(os.path.join(out_path, allnames))
        np.save(os.path.join(out_path, allnames, 'CT.npy'), img1_numpy)
        np.save(os.path.join(out_path, allnames, 'PET.npy'), img2_numpy)

    except:
        print(allnames,'有问题')
