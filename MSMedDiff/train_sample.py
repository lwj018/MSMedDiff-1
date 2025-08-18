import os
import random

import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from dice_loss.dice_loss import *
from multi_train_utils.distributed_utils import is_main_process
from utils import *
import einops
import logging
import timm

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        # print(x.shape)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        # sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        # print(sqrt_alpha_hat.shape)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        # print(sqrt_one_minus_alpha_hat.shape)
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_condition_decoder(self, model, n, image, in_ch=1, is_train=True):
        logging.info(f"Sampling {n} new images....")
        if not is_train:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model.eval()

        with torch.no_grad():

            x = torch.randn((n, in_ch, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                # predicted_noise = model(torch.unsqueeze(x,dim=1), t,images)
                # print(image.shape,x.shape)
                predicted_noise = model(x, t, image)
                # predicted_noise = torch.squeeze(predicted_noise)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
                # x = x.clamp(-1.0,1.)
        if is_train:
            model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x


def train(device, model, dataloader, optimizer, diffusion, epoch, image_size=256):
    # mse = nn.MSELoss()
    #s_loss = CombinedLoss()
    mae_loss = MAELoss()
    #smooth_loss = SmoothLoss()
    #dice_loss = DiceLoss()
    #hmae_loss=MAELossWithHessian()
    # bce = nn.()
    # diceloss = DiceLoss()
    loss_mean = 0.0
    model.train()
    if is_main_process():
        dataloader = tqdm(dataloader, colour="green", ncols=80)
    for i, (images, lable) in enumerate(dataloader):
        images = images.to(device)
        lable = lable.to(device)
        # with autocast():
            # images,lable=images,lable
            # images = images.to(device)
            # lable = lable.to(device)

        t = diffusion.sample_timesteps(lable.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(lable, t)
            # print(x_t.shape,noise.shape)
        predicted_noise = model(x_t, t, images)
            # print(noise.shape,predicted_noise.shape)
            # loss = mse(noise, predicted_noise)
            # 对模型输出进行sigmoid激活
            # predicted_noise = (predicted_noise.clamp(-1, 1) + 1) / 2
            # print(set(predicted_noise.reshape(-1).tolist()))
            # print(set(noise.reshape(-1).tolist()))
            #loss = mae_loss(predicted_noise, noise)
            # print(predicted_noise.shape)
            #loss = mae_loss(predicted_noise, noise)  # 添加mix_loss
            # if epoch>400:
            #     loss=alpha * mae_loss(predicted_noise, noise) + (1 - alpha) * s_loss(predicted_noise, noise)
            # else:
        loss=mae_loss(predicted_noise, noise)
            # loss=ifl(predicted_noise, noise)
        optimizer.zero_grad()

        # scalar.scale(loss).backward()
        # scalar.step(optimizer)
        # scalar.update()
        loss.backward()
        optimizer.step()
        loss_mean = loss_mean + loss.item()
        if is_main_process():
            dataloader.desc = "epoch: {} loss_mean: {}".format(epoch, round(loss_mean / (i + 1), 3))

    return loss_mean
def val(device, model, dataloader, diffusion, epoch):
    # mse = nn.MSELoss()
    #s_loss = CombinedLoss()
    mae_loss = MAELoss()
    model.eval()
    loss_mean = 0.0
    if is_main_process():
        dataloader = tqdm(dataloader, colour="green", ncols=80)
    with torch.no_grad():
        for i, (images, lable) in enumerate(dataloader):
            images = images.to(device)
            lable = lable.to(device)
            # with autocast():
                # images,lable=images,lable
                # images = images.to(device)
                # lable = lable.to(device)

            t = diffusion.sample_timesteps(lable.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(lable, t)

            predicted_noise = model(x_t, t, images)

            loss=mae_loss(predicted_noise, noise)

            loss_mean = loss_mean + loss.item()
            if is_main_process():
                dataloader.desc = "epoch: {} loss_mean: {}".format(epoch, round(loss_mean / (i + 1), 3))

        return loss_mean
# def train_vit(device, model, dataloader, optimizer, diffusion, epoch):
#     model = timm.create_model('vit_base_patch16_224', pretrained=True)
#     # 固定除最后一层以外的所有层的参数
#     for param in model.parameters():
#         param.requires_grad = False
#     # 替换最后一层为新的线性层
#     model.fc = nn.Linear(in_features=768, out_features=num_classes)
#     # 训练最后一层
#     optimizer.zero_grad()
#     for i, (images, lable) in enumerate(dataloader):
#         # print(images.shape,lable.shape)
#         images = images.to(device)
#
#         lable = lable.to(device)
#         outputs = model(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
