import argparse
from skimage import io
import os
import numpy as np
import math
import itertools
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
        #修改代码 只扩大er倍
        for out_features in range(1):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                #尺度变换4*4 选用基于插值bicubic的Upsample替代PixelShuffle
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
class C(nn.Module):
    def __init__(self, input_shape):
        super(C, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def c_block(in_filters, out_filters, first_block=False):
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
            layers.extend(c_block(in_filters, out_filters, first_block=(i == 0)))
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
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.files = sorted(glob.glob(root + "/*.*"))
    def __getitem__(self, index):
        img = io.imread(self.files[index % len(self.files)])
        img_ts = self.lr_transform(img)
        img_ts = (img_ts - 304) / ((2335 - 304) / 2) - 1
        w = img_ts.shape[2]//2
        img_lr = img_ts[:, :,w:]
        img_hr = img_ts[:, :,:w]
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

DATBASE_PATH = r'C:/Users/HP/uio.prog/MSc/GISR_code'
epoch=0        #epoch to start training from
n_epochs=100        #number of epochs of training
dataset_name="data"      #name of the dataset
batch_size=16    #size of the batches
lr=0.0002         #adam: learning rate
b1=0.5      #decay of first order momentum of gradient
b2=0.999    #help="adam: decay of first order momentum of gradient
decay_epoch=100      #help="epoch from which to start lr decay
n_cpu=0   #help="number of cpu threads to use during batch generation
hr_height=64    #high resolution image height
hr_width=64      #high resolution image width
channels=1         #number of image channels
sample_interval=100              #interval between saving image samples")
checkpoint_interval=10         #interval between model checkpoints

cuda = torch.cuda.is_available()
hr_shape = (hr_height, hr_width)
generator = GeneratorResNet()
c= C(input_shape=(channels, *hr_shape))
feature_extractor = FeatureExtractor()

feature_extractor.eval()

criterion_r=RMSELoss()

criterion_content = torch.nn.L1Loss()


if cuda:
    generator = generator.cuda()
    c = c.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_r = criterion_r.cuda()
    criterion_content = criterion_content.cuda()

if epoch != 0:
    generator.load_state_dict(torch.load(DATBASE_PATH+"/model/generator_90.pth"))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_C = torch.optim.Adam(c.parameters(), lr=lr*0.1, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset(DATBASE_PATH+"/"+"%s" % dataset_name, hr_shape=hr_shape),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)
writer = SummaryWriter(DATBASE_PATH+'/logs')




for epoch in range(epoch, n_epochs):
    start = time.time()
    for i, imgs in enumerate(dataloader):

        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *c.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *c.output_shape))), requires_grad=False)
        optimizer_G.zero_grad()
        gen_hr = generator(imgs_lr)
        gen_hr = (gen_hr + imgs_lr)/2
        loss_r = criterion_r(c(gen_hr), valid)
        real_dem_3c = torch.cat((imgs_hr,imgs_hr,imgs_hr),1)
        gen_dem_3c = torch.cat((gen_hr, gen_hr, gen_hr), 1)

        gen_features = feature_extractor(gen_dem_3c)
        real_features = feature_extractor(real_dem_3c)
        loss_content = criterion_content(gen_features, real_features.detach())
        loss_RMSE = criterion_r(imgs_hr, gen_hr)

        loss_G = loss_content + 1e-3 * loss_r+10*loss_RMSE
        loss_G.backward()
        optimizer_G.step()

        optimizer_C.zero_grad()

        loss_real = criterion_r(c(imgs_hr), valid)
        loss_fake = criterion_r(c(gen_hr.detach()), fake)

        loss_C = (loss_real + loss_fake) / 2
        if i%3==0:
            loss_C.backward()
            optimizer_C.step()
        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d]  [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader),loss_G.item())
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            writer.add_scalar('loss_G', loss_G.item(), batches_done)
            writer.add_scalar('loss_RMSE', loss_RMSE.item(), batches_done)
            img_grid = torch.cat((imgs_lr, gen_hr,imgs_hr), -1)
            save_image(img_grid, DATBASE_PATH+"/img/%d.png" % batches_done, normalize=False)
    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1 ,
                                                           time.time() - start))

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), DATBASE_PATH+"/model/generator_%d.pth" % epoch)