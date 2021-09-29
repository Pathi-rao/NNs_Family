import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Conv2d, Module, BatchNorm2d, Dropout, MaxPool2d
from torch.nn.modules import batchnorm
from torchsummary import summary

""" we add Batch Normalization after the activation function of the output layer or before the activation function of the input layer. 
Mostly researchers found good results in implementing Batch Normalization after the activation layer.
"""

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*94*94, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Current device: ", device)

# model = SiameseNetwork()
# model = model.cuda()
# # model = model.to('cuda:0') # send the model to device cause sometimes there are errors regarding device

# print(summary(model, [(1, 100,100), (1, 100,100)]))