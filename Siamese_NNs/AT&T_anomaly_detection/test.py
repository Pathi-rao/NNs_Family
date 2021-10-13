from utils import imshow, show_plot
import random
import numpy as np
from PIL import Image
import PIL.ImageOps
import random

import torch
import torchvision
from torch.utils.data import DataLoader

import datahandler as dh

root_dir = '..\..\..\Dataset\AT&T_Face'

""" The top row and the bottom row of any column is one pair. The 0s and 1s correspond to the column of the image. 
    1 indiciates dissimilar, and 0 indicates similar. """

if __name__ == '__main__': # to avoid multi-processing error

    siamese_train_dataset, _, _ = dh.pre_preocessor(root_dir=root_dir, trainbatchsize=8, testbatchsize=8)

    vis_dataloader = DataLoader(siamese_train_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=8)
    dataiter = iter(vis_dataloader)
    # print(dataiter)

    example_batch = next(dataiter)
    print(example_batch)
    # concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    # print(example_batch[2].numpy())
    # imshow(torchvision.utils.make_grid(concatenated))
    

