import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import random
import torch.nn.init as init
import sys
import argparse
import pickle
import lasagne
import psutil
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner   

def get_memory():
    mem = psutil.virtual_memory()
    return mem.used
before_memory = get_memory()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
from models import * 


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_validation_data(data_folder, mean_image, img_size=32):
    test_file = os.path.join(data_folder, 'val_data')

    d = unpickle(test_file)
    x = d['data']
    y = d['labels']
    x = x / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = np.array([i-1 for i in y])

    # Remove mean (computed from training data) from images
    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return dict(
        X_test=x,
        Y_test=y.astype('int32') )

def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')
    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)
    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        mean=mean_image)



model_test = resnet18('ImageNet').to(device)
optimizer = optim.SGD(params=model_test.parameters(), lr=0.01)

img_size = 32
batch_range = list(range(1, 11))
random.shuffle(batch_range)
traindir = 'data/ImageNet_train'
valdir = 'data/ImageNet_val'
imnetdata = load_databatch(traindir, 1, img_size=img_size)
mean_image = imnetdata['mean']
imnetdata = load_validation_data(valdir, mean_image=mean_image, img_size=img_size)
X_test = imnetdata['X_test']
Y_test = imnetdata['Y_test']
del imnetdata
test_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(Y_test))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False,
                                            num_workers=6, pin_memory=True)
del X_test, Y_test

for i in tqdm(range(50)):
    for index in range(0,10):
        print("--------------",index,'data begin-----------------')
        ib = batch_range[index]
        imnetdata = load_databatch(traindir, ib, img_size=img_size)
        X_train_prune = imnetdata['X_train']
        Y_train_prune = imnetdata['Y_train']
        mean_image = imnetdata['mean']
        del imnetdata
        train_dataset = torch.utils.data.TensorDataset(
            torch.cat([torch.FloatTensor(X_train_prune), torch.FloatTensor(X_train_prune[:,:,:,::-1].copy())], dim=0),
            torch.cat([torch.LongTensor(Y_train_prune), torch.LongTensor(Y_train_prune)], dim=0))
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6, pin_memory=True)
        del X_train_prune, Y_train_prune
        
        model_test.train()
        for input, target in train_loader:
            input = input.cuda(non_blocking=True)
            target= target.cuda(non_blocking=True)
            optimizer.zero_grad()
            output,_ = model_test(input)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    model_test.eval()
    correct = 0
    for input, target in test_loader:
        input = input.cuda(non_blocking=True)
        target= target.cuda(non_blocking=True)
        output,_ = model_test(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()   
    print(correct/len(test_loader.dataset))

    torch.save(model_test.state_dict(), 'ResNet18_for_LightCL.pth')
