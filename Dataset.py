from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image
import torch
import PIL.ImageOps
import os

class STAMDataset(Dataset):

    """
    based on : https://www.cnblogs.com/inchbyinch/p/12116339.html
    """
    def __init__(self, image_folder_dataset,image_folder_dataset_night, transform=None, should_invert=False):
        self.imageFolderDataset = image_folder_dataset
        self.night_image = image_folder_dataset_night
        self.transform = transform
        self.should_invert = should_invert
        image_dictionary = {}
        for img_tuple in image_folder_dataset.imgs:
            key = img_tuple[1]
            if key in image_dictionary.keys():
                image_dictionary[key].append(img_tuple)
            else:
                image_dictionary[key] = [img_tuple]
        self.image_dictionary = image_dictionary

    def __getitem__(self, index):
        image_dictionary_num = len(self.image_dictionary)
        randint = random.randint(0, image_dictionary_num - 1)
        y1 = randint
        index1 = random.randint(0, len(self.image_dictionary[randint]) - 1)
        img0_tuple = self.image_dictionary[randint][index1]

        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        
        img1 = Image.open(img0_tuple[0])
        img2 = Image.open(img1_tuple[0])

        img1_night = os.path.basename(img0_tuple[0])
        img2_night = os.path.basename(img1_tuple[0])

        img1_night = self.night_image+ '/'+img1_night
        img2_night = self.night_image + '/'+img2_night

    
        img3 = Image.open(img1_night)
        img4 = Image.open(img2_night)
       


        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)


        return img1, img2, img3,img4, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)),img0_tuple[1],img1_tuple[1]

        # return img1, img2, img3,img4, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))  
 

    def __len__(self):
        return len(self.imageFolderDataset.imgs)