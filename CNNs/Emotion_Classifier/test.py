'''
This file is only for testing things
    '''
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import datahandler as dh

#pre_preocessor function takes absolute path and batch_size as arguments
batch_size = 2
train_loader, test_loader = dh.pre_processor(
                                'C:\\Users\\User\\Desktop\\Strive_School\\Github\\Datasets\\archive\\fer2013\\',
                                batch_size)

# print(train_loader)
# print(test_loader)

classes = ('Angry', 'Disgust', 'Fear', 'Happy',
            'Neutral', 'Sad', 'Surprise')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size))) #2 is the batch_size


"""
    If you run  the script till this point, it will return 3 outputs. 
    1. displays the images
    2. Torch size such as ([2 ,1 ,48, 48)] where 2 is the batch size of images that are being loaded,
        1 is the grayscale images and 48*48 is the image size.
    3. Classes of the images
    """

# alternative way to test the dataloader and display images
# def imshow(image, ax=None, title=None, normalize=False):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()
#     image = image.numpy().transpose((1, 2, 0))

#     if normalize:
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         image = std * image + mean
#         image = np.clip(image, 0, 1)

#     ax.imshow(image)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.tick_params(axis='both', length=0)
#     ax.set_xticklabels('')
#     ax.set_yticklabels('')

#     return ax
# # Run this to test your data loaders
# images, labels = next(iter(train_loader))
# imshow(images[0], normalize=False)