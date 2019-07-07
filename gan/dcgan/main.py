import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = 'datasets/celeba'
workers = 4
batch_size = 128

image_size = 64
nc = 3
# Size of z latent vector
nz = 100
# Size of feature maps in generator
ngf = 64
#Size of feature maps in discriminator
ndf = 64

epochs = 5
lr = 0.0002
beta1 = 0.5
n_gpu = 1

dataset = 