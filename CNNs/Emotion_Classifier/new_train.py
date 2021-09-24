from model import CNN
import datahandler as dh
import matplotlib.pyplot as plt
import numpy as np
import sys

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torchsummary import summary
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

train_loader, test_loader = dh.pre_processor(
                                'C:\\Users\\User\\Desktop\\Strive_School\\Github\\Datasets\\archive\\fer2013\\',
                                batchsize = 32)

writer = SummaryWriter("runs/emotion_classifier")

classes_ = ('Angry', 'Disgust', 'Fear', 'Happy',
            'Neutral', 'Sad', 'Surprise')

# helper function to show images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# the tensorboard will display images w.r.t the batch size
dataiter = iter(train_loader)
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image("Emotion_images", img_grid)
writer.close()
# sys.exit()



train_losses = []
val_losses = []
val_loss = []


def train(epochs, train_loader, model, optimizer, criterion, scheduler, best_val, print_every = 40):
    running_correct = 0
    model.train()

    for each_epoch in range(epochs):

        tr_loss = 0

        for i , (images, labels) in enumerate(iter(train_loader)):
            # print(images.shape())
            # images = images.reshape(-1, 48*48).to(device)
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            output_train = model(images)
            loss_train = criterion(output_train, labels)

            loss_train.backward()
            optimizer.step()
            
            tr_loss += loss_train.item()

            _, predicted = torch.max(output_train.data, 1)
            running_correct += (predicted == labels).sum().item()


    #     if i % print_every == 0:
    #         # print(f"\tIteration: {i}\t Loss: {tr_loss/print_every:.4f}")
    #         train_losses.append(tr_loss/images.shape[0])
    #         print('Epoch : ',each_epoch, "\t Train loss: ", tr_loss/images.shape[0])

    #         val_loss = 0
    #         with torch.no_grad():
    #             for i , (val_images, val_labels) in enumerate(iter(test_loader)):
    #                 output = model(val_images.to(device))
    #                 loss_val = criterion(output, val_labels.to(device))
    #                 val_loss += loss_val.item()


    #         val_loss =  val_loss/val_images.shape[0]
    #         val_losses.append(val_loss)
    #         print('Epoch : ',each_epoch, "\t Train loss: ", tr_loss/val_images.shape[0], "\t Validation loss: ", val_loss)

    #         if val_loss < best_val:
    #             best_val = val_loss
    #             torch.save(model, "./models/best_model.pth")
    #     scheduler.step()
    # plt.plot(train_losses)
    # plt.plot(val_losses)
    # plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: ", device)

model = CNN().to(device)

# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

# defining the loss function
criterion = nn.CrossEntropyLoss()


# summary(model,(1, 48, 48)) # since the input images are of gray sclae size 48*48

# train(epochs = 5, train_loader = train_loader, model = model, optimizer = optimizer,
#         criterion = criterion, scheduler = scheduler, best_val = 2, print_every = 40)