import rasterio
import argparse
from skimage import io
import os
import numpy as np
import math
import itertools
import tifffile as tiff
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.models.vgg import VGG19_Weights
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
    def forward(self, img):
        return self.feature_extractor(img)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(1):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                #nn.Upsample(scale_factor=2, mode='bicubic'),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=2, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
class D(nn.Module):
    def __init__(self, input_shape):
        super(D, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def d_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(d_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
    def forward(self, pred, truth):
        return  torch.sqrt(torch.mean((pred-truth)**2))
        torch.square((pred-truth)**2)

path = 'C:/Users/HP/uio.prog/MSc/GISR_code' ##### Your folder path ####
# sample = "R1" #### Your file name of tiff ####
sample = "bathy_fps11-15-3_miniAOI_superResTest" #### Your file name of tiff ####
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
lr = 0.0002
b1 = 0.5
b2 = 0.999
hr_height = 64
hr_width = 64
h=2335
l=304
cuda = torch.cuda.is_available()
hr_shape = (hr_height, hr_width)
generator = GeneratorResNet()
feature_extractor = FeatureExtractor()
feature_extractor.eval()
criterion_GAN = RMSELoss()
criterion_content = torch.nn.L1Loss()
if cuda:
    generator = generator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
generator.load_state_dict(torch.load(path+'/model/generator_90.pth'))    ## GPU-version ##
# generator.load_state_dict(torch.load(path+'/model/generator_90.pth', map_location=torch.device('cpu'))) ## CPU-version ##
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
a = tiff.imread(path+'/image/'+sample+'.tif')
a1 = a[:, 64:]
a111 = a[:,:64]
a2 = (a1) / ((2335 - 304) / 2) - 1
a3 = np.expand_dims(a2, 0)
a4 = np.expand_dims(a3, 0)
a5 = a4.astype(np.float32)
a6 = torch.tensor(a5)
Tensor = torch.cuda.FloatTensor
a7 = Variable(a6.type(Tensor))
a8 = generator(a7)
a9=(a8+a7)/2
a10 = (a9+1)*((h - l) / 2)
a11 = a10.cpu().detach().numpy()
# save_image(a10[0, 0, :, :], path +'/'+'GISR_' + sample + '.png', normalize=True)

# Original export
tiff.imwrite(path +'/'+'GISR_' + sample + '.tif', a11[0,0,:,:])

# Export nr. 2 - attempt to fix CRS with rasterio
a11 = a11[0, 0, :, :]
output_path = path + '/' + 'GISR_' + sample + '.tif'

# Read the original image's CRS and transform properties
with rasterio.open(path+'/image/'+sample+'.tif') as src:
    src_crs = src.crs
    src_transform = src.transform

# Save the output image with the same CRS and transform properties
with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=a11.shape[0],
    width=a11.shape[1],
    count=1,
    dtype=a11.dtype,
    crs=src_crs,
    transform=src_transform,
) as dst:
    dst.write(a11, 1)
