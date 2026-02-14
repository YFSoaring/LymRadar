import os
import random
import torchio as tio
import numpy as np
from torch.utils.data import Dataset
import albumentations as A


class myNumpyDatasetNolabel(Dataset):
    def __init__(self, root_dir, img_list1,img_list2, transform):
        self.root_dir = root_dir
        self.img_list1 = img_list1
        self.img_list2 = img_list2
        self.transform = transform

    def normalize(self,image):
        mean = np.mean(image)
        var = np.mean(np.square(image - mean))
        image = (image - mean) / np.sqrt(var)
        return image

    def transform_train(self,image):
        train_trans = tio.Compose([
            tio.RandomFlip(flip_probability=0.5),  # 随机翻转
            tio.RandomAffine(  # 随机仿射
                scales=(0.8, 1.2),  # 指定缩放比例
                degrees=10,  # # 旋转的角度
                translation=0,
                image_interpolation='nearest'),  # 平移的角度
            tio.RandomAnisotropy(
            axes=(0, 1, 2),
            downsampling=(2, 5),),
            tio.RandomGamma(log_gamma=(-0.3, 0.3)),
        ])
        image = train_trans(image)
        return image


    def __len__(self):
        return len(self.img_list1)

    def __getitem__(self, idx):
        img_label1, img_path1  = self.img_list1[idx]
        img_label2, img_path2 = self.img_list2[idx]

        if self.transform == "test":
            # read image
            img_name1 = img_path1
            img_name2 = img_path2
            img_id = img_path1.split('/')[-2]
            img_array1 = np.load(img_name1)
            img_array2 = np.load(img_name2)
            img_array1 = self.normalize(img_array1)
            img_array2 = self.normalize(img_array2)
            img_array = np.concatenate((img_array1, img_array2), axis=0)
            img_array = np.expand_dims(img_array, axis=0)
            midpoint = img_array.shape[1] // 2
            new_img_array1 = img_array[:,:midpoint, :, :]
            new_img_array2 = img_array[:,midpoint:, :, :]
            new_img_array1 = np.array(new_img_array1, dtype=np.float32)
            new_img_array2 = np.array(new_img_array2, dtype=np.float32)

            return new_img_array1, new_img_array2, img_id

class myNumpyDataset(Dataset):
    def __init__(self, root_dir, img_list1,img_list2, transform):

        self.root_dir = root_dir
        self.img_list1 = img_list1
        self.img_list2 = img_list2
        self.transform = transform

    def normalize(self,image):
        mean = np.mean(image)
        var = np.mean(np.square(image - mean))
        image = (image - mean) / np.sqrt(var)
        return image

    def transform_train(self,image):
        train_trans = tio.Compose([
            tio.RandomFlip(flip_probability=0.5),  # 随机翻转 0.3的概率翻转
            tio.RandomAffine(  # 随机仿射
                scales=(0.8, 1.2),  # 指定缩放比例
                degrees=10,  # # 旋转的角度
                translation=0,
                image_interpolation='nearest'),  # 平移的角度
            tio.RandomAnisotropy(
            axes=(0, 1, 2),
            downsampling=(2, 5),),
            tio.RandomGamma(log_gamma=(-0.3, 0.3)),
        ])
        image = train_trans(image)
        return image

    def __len__(self):
        return len(self.img_list1)

    def __getitem__(self, idx):
        img_label1, img_path1  = self.img_list1[idx]
        img_label2, img_path2 = self.img_list2[idx]

        if self.transform == 'train':
            # read image and labels
            img_name1 = img_path1
            label_name1 = img_label1
            img_name2 = img_path2
            label_name2 = img_label2
            try:
                assert os.path.isfile(img_name1)
            except:
                print(img_name1,'有问题')
            try:
                assert os.path.isfile(img_name2)
            except:
                    print(img_name2, '有问题')
            img_array1 = np.load(img_name1)
            img_array2 = np.load(img_name2)

            img_array1 = self.normalize(img_array1)
            img_array2 = self.normalize(img_array2)
            img_array = np.concatenate((img_array1,img_array2),axis=0)
            img_array = np.expand_dims(img_array, axis=0)  # 用于在指定的位置添加新的维度
            img_array = self.transform_train(img_array)

            ###
            midpoint = img_array.shape[1] // 2
            new_img_array1 = img_array[:,:midpoint, :, :]
            new_img_array2 = img_array[:,midpoint:, :, :]

            new_img_array1 = np.array(new_img_array1, dtype=np.float32)
            new_img_array2 = np.array(new_img_array2, dtype=np.float32)

            return new_img_array1, label_name1, new_img_array2, label_name2

        elif self.transform == "test":
            img_name1 = img_path1
            label_name1 = img_label1
            img_name2 = img_path2
            label_name2 = img_label2
            img_id = img_path1.split('/')[-2]
            img_array1 = np.load(img_name1)
            img_array2 = np.load(img_name2)
            img_array1 = self.normalize(img_array1)
            img_array2 = self.normalize(img_array2)
            img_array = np.concatenate((img_array1, img_array2), axis=0)
            img_array = np.expand_dims(img_array, axis=0)  # 用于在指定的位置添加新的维度
            midpoint = img_array.shape[1] // 2
            new_img_array1 = img_array[:,:midpoint, :, :]
            new_img_array2 = img_array[:,midpoint:, :, :]
            new_img_array1 = np.array(new_img_array1, dtype=np.float32)
            new_img_array2 = np.array(new_img_array2, dtype=np.float32)

            return new_img_array1, label_name1, new_img_array2, label_name2, img_id
