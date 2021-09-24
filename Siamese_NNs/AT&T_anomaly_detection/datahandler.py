import numpy as np
from PIL import Image
import PIL.ImageOps
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


training_dir = '..\Dataset\AT&T_Face\Train'
testing_dir = '..\Dataset\AT&T_Face\Test'

class SiameseNetworkDataset(Dataset):
    
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 

        if should_get_same_class: # Enter the loop if value is 1
            # while True:
            for i in range(0, 1):
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            # while True:
            for i in range(0, 1):
                #keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        # converts an image to grayscale if passed "L". 
        # read more here --> https://stackoverflow.com/questions/52307290/what-is-the-difference-between-images-in-p-and-l-mode-in-pil
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            """ 
            The ImageOps module contains a number of 'ready-made' image processing operations
            convert series of images drawn as white on black background images to 
                images where white and black are inverted (as negative)
            """
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0) # apply the given tranformation(when instantiating the class).
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)



def pre_preocessor():

    folder_dataset = datasets.ImageFolder(root=training_dir)
    # print(folder_dataset.imgs) 
    # o/p will be tuple with each image path and the folder index
    # [('..\\..\\..\\Dataset\\AT&T_Face\\Train\\s1\\1.pgm', 0), ('..\\..\\..\\Dataset\\AT&T_Face\\Train\\s1\\10.pgm', 0)....]  

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset = folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100,100)),
                                            # we can also use custom collate function to resize with padding if have different resolution
                                                                            transforms.ToTensor()
                                                                            ])
                                            ,should_invert=True)

    return siamese_dataset