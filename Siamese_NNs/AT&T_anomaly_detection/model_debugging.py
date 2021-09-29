import torch
import torch.nn as nn
from torch.nn.modules import activation
from torchsummary import summary

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.size())
        return x

class SiameseNetwork(nn.Module):
    
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.reflectivepad1 = nn.ReflectionPad2d(1),
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3),
        self.batchnorm1 = nn.BatchNorm2d(4),
        self.relu1 = nn.ReLU(inplace=True),

        self.reflectivepad2 = nn.ReflectionPad2d(1),
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3),
        self.batchnorm2 = nn.BatchNorm2d(8),
        self.relu2 = nn.ReLU(inplace=True),

        self.reflectivepad3 = nn.ReflectionPad2d(1),
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3),
        self.batchnorm3= nn.BatchNorm2d(8),
        self.relu3 = nn.ReLU(inplace=True),

        self. fc1 = nn.Linear(8*94*94, 500),
        self.relu4 = nn.ReLU(inplace=True),

        self.fc2 = nn.Linear(500, 500),
        self.relu5 = nn.ReLU(inplace=True),

        self.fc3 = nn.Linear(500, 5)
        # Print()


    def forward(self, x):

        """
        -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        """

        x = self.relu1(self.batchnorm1(self.conv1((self.reflectivepad1(x)))))

        x = self.relu2(self.batchnorm2(self.conv2((self.reflectivepad2(x)))))

        x = self.relu3(self.batchnorm3(self.conv3((self.reflectivepad3(x)))))

        x = x.view(-1, 8*94*94)

        x = self.relu4(self.fc1(x))

        x = self.relu5(self.fc2(x))

        x = self.fc3(x)

        return x

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation [name] = output.detach()
    return hook


model = SiameseNetwork()
model.fc3.register_forward_hook(get_activation('fc1'))
x = torch.randn(1, 25)
output = model(x)
print(activation['fc1'])