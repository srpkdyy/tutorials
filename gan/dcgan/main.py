from __future__ import print_function
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
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def im_show(batch):
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Training Images')
    images = vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True)
    images = np.transpose(images.cpu(), (1, 2, 0))
    plt.imshow(images)
    plt.show()
    

def main():
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    dataroot = 'data/celeba'
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
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=dataroot, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    device = torch.device('cuda' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')
    
    # Use if you want to confirm training images
    #real_batch = next(iter(data_loader))
    #im_show(real_batch)

    netG = Generator(nc, nz, ngf, n_gpu).to(device)
    netD = Discriminator(nc, nz, ndf, n_gpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (n_gpu > 1):
        device_ids = list(range(n_gpu))
        netG = nn.DataParallel(netG, device_ids)
        netD = nn.DataParallel(netD, device_ids)

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netG.apply(weights_init)
    netD.apply(weights_init)

    print(device)
    print(netG)
    print(netD)

    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    optimizersD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizersG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    

class Generator(nn.Module):
    def __init__(self, nc, nz, ngf, n_gpu):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            #input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), 
            nn.ReLU(True),
            #state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.mai(input)


class Discriminator(nn.Module):
    def __init__(self, nc, nz, ndf, n_gpu):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    main()
