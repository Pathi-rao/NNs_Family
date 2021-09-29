from utils import imshow, show_plot

import torch
from torch import optim
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

import datahandler as dh
from model import SiameseNetwork
from contrastive_loss import ContrastiveLoss

dataset_root_dir = '..\..\..\Dataset\AT&T_Face'
epochs = 100
train_batchsize = 64
test_batchsize = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_loader, test_loader = dh.pre_preocessor(root_dir = dataset_root_dir, trainbatchsize = train_batchsize, 
                                                testbatchsize = test_batchsize)


# load the model
net = SiameseNetwork().cuda()

criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0005 )


def train():
    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0, epochs):
        for i, data in enumerate(train_loader, 0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()

            # reset the gradients
            optimizer.zero_grad()

            output1, output2 = net(img0, img1)                      # forward pass
            loss_contrastive = criterion(output1, output2, label)   # compute loss

            loss_contrastive.backward()                             # backward pass
            optimizer.step()                                        # update model

            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    return counter, loss_history

counter, loss_history = train()

show_plot(counter, loss_history)

# testing
dataiter = iter(test_loader)
x0, _, _ = next(dataiter)

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))