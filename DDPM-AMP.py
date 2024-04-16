#!/usr/bin/env python
# coding: utf-8

# # Initialize

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torch import einsum
from einops import rearrange
from tqdm import tqdm
from PIL import Image, ImageFile
from pickle import load, dump
import math
import random
import cv2
import time
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class Util:
    @staticmethod
    def extract(v, t, shape):
        out = torch.gather(v, index=t, dim=0).float().to(t.device)
        out = out.view([t.size(0)] + [1] * (len(shape) - 1)) # (Batch, 1, 1, 1, ...)
        return out
    
    @staticmethod
    def loadImages(batch_size, path, size):
        images = ImageFolder(path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            transforms.RandomCrop(size),
            transforms.ToTensor()
        ]))
        return DataLoader(images, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count(), pin_memory=True)


# # Denoising Diffusion Probabilistic Model (DDPM)

# ## 拡散過程
# $\epsilon_t 〜 N(0, I)\quad(I: 単位行列)$ とし、
# $$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_tt\quad(\beta: ノイズの大きさのハイパーパラメータ)$$
# とするマルコフ過程を考えると、 
# $$q(x_t | x_{t-1}) = N(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_tI)$$
# $$∴x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1 - \bar{\alpha_t}}\epsilon\quad(\alpha_t = 1 - \beta_t, \bar{\alpha} = \prod_{t=1}(\alpha_t))$$
# また、$q(x_t | x_0) = N(x_t; \sqrt{\bar{\alpha}}x_0, \sqrt{1 - \bar{\alpha}}I)$

# ## 条件付き逆拡散過程
# ベイズの定理とマルコフ性より、
# $$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1})q(x_{t-1} | x_0)}{q(x_t | x_0)}$$
# それぞれの条件付き確率を拡散過程の正規分布より求めると、
# $$q(x_{t-1} | x_t, x_0) ∝ exp[-\frac{1}{2}\frac{(x_{t-1} - \bar{\mu}_t(x_t, x_0))^2}{\bar{\beta}_t}]$$
# $$∴q(x_t | x_t, x_0) = N(x_{t-1}; \bar{\mu}_t(x_t, x_0), \bar{\beta}_tI)$$
# また、$$\bar{\mu}_t = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon)　…①$$
# $$\bar{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t$$
# ここで、$\sigma_t^2 = \bar{\beta}_t$とし、
# $$x_{t-1} = \bar{\mu}_t + \bar{\beta}_tz_t\quad(z_t 〜 N(0, I))$$
# $$= \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t) + \sigma_t^2z_t$$
# $$= \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1 - α_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t) + \bar{\beta}z_t$$

# ## 損失関数
# ニューラルネットワークのパラメータをθとすると、
# $$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t^2z_t\quad(z_t 〜 N(0, I))$$
# ここで、画像生成のため、平均の負の対数尤度を考えると、\
# マルコフ性と正規分布のカルバックライブラーダイバージェンスより、
# $$L_t = E_q[\frac{1}{2\sigma_t^2} || \bar{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) ||^2]$$
# ①より、$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta)$と定義し、係数を簡略化すると、
# $$L_t^{simple} = E_q[||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t, t) ||^2]$$

# 参考: [拡散モデルの基礎と研究事例: Imagen](https://qiita.com/iitachi_tdse/items/6cdd706efd0005c4a14a) \
# 参考: [DenoisingDiffusionProbabilityModel-ddpm-](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-)

# In[ ]:


# DDPM - 拡散過程
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_t, num_times):
        super().__init__()

        self.model = model
        self.num_times = num_times

        betas = nn.Parameter(torch.linspace(beta_1, beta_t, num_times).double())
        
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.sqrt_alphas_bar = nn.Parameter(torch.sqrt(alphas_bar))
        self.sqrt_one_minus_alphas_bar = nn.Parameter(torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        t = torch.randint(self.num_times, size=(x_0.size(0),), device=x_0.device)
        noise = torch.randn_like(x_0)
        
        x_t = (Util.extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
               + Util.extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


# In[ ]:


# DDPM - 条件付き逆拡散過程
class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_t, num_times):
        super().__init__()
        
        self.model = model
        self.num_times = num_times
        
        betas = torch.linspace(beta_1, beta_t, num_times).double()
        
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:num_times]
        
        self.coeff1 = nn.Parameter(torch.sqrt(1. / alphas))
        self.coeff2 = nn.Parameter(self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        
        self.var = nn.Parameter((1. - alphas_bar_prev) / (1. - alphas_bar) * betas)

    def p_mean_variance(self, x_t, t):
        eps = self.model(x_t, t)
        xt_prev_mean = (Util.extract(self.coeff1, t, x_t.shape) * x_t
                        - Util.extract(self.coeff2, t, x_t.shape) * eps)
        var = Util.extract(self.var, t, x_t.shape)
        return xt_prev_mean, var

    def forward(self, x):
        xt_shape = x.shape
        for times in reversed(range(self.num_times)):
            t = torch.ones([xt_shape[0]], dtype=torch.long, device=x.device) * times
            x = x.detach()
            t = t.detach()
            mean, var= self.p_mean_variance(x, t)
            if times > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            x = mean + torch.sqrt(var) * noise
            assert torch.isnan(x).int().sum() == 0, "NaN in Tensor."
        return torch.clip(x, 0, 1)


# # UNet

# In[ ]:


class TimeEmbedding(nn.Module):
    def __init__(self, num_times, dim_in, dim_out):
        super().__init__()
        assert dim_in % 2 == 0
        
        embedding = torch.arange(0, dim_in, step=2) / dim_in * math.log(10000)
        embedding = torch.exp(-embedding)
        position = torch.arange(num_times).float()
        embedding = position.unsqueeze(0).T * embedding
        embedding = torch.stack([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        embedding = embedding.view(num_times, dim_in)
        
        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(embedding),
            nn.Linear(dim_in, dim_out),
            nn.Mish(inplace=True),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, t):
        return self.embedding(t)


# In[ ]:


class SelfAttention(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        
        # Pointwise Convolution
        self.query_conv = nn.Conv2d(dim_in, dim_in // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(dim_in, dim_in // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(dim_in, dim_in, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, return_map=False):
        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        s = torch.bmm(proj_query, proj_key)
        attention_map_T = F.softmax(s, dim=-2)
        
        proj_value = self.value_conv(x).view(x.size(0), -1, x.size(2) * x.size(3))
        o = torch.bmm(proj_value, attention_map_T)
        
        o = o.view(x.size(0), x.size(1), x.size(2), x.size(3))
        out = x + self.gamma * o
        
        if return_map:
            return out, attention_map_T.permute(0, 2, 1)
        else:
            return out


# In[ ]:


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_t, dropout=0., attention=False, num_groups=32):
        super().__init__()

        if dim_in != dim_out:
            self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        
        self.residual_1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        self.time_projection = nn.Linear(dim_t, dim_out)
        self.residual_2 = nn.Sequential(
            nn.GroupNorm(num_groups, dim_out),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, dim_out)
        )
        if attention:
            self.attention = SelfAttention(dim_out)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        shortcut = self.shortcut(x)
        
        residual_1 = self.residual_1(x)
        time_projection = self.time_projection(t)[:, :, None, None]
        
        residual_2 = self.residual_2(residual_1 + time_projection)
        
        residual = F.mish(residual_2 + shortcut)
        out = self.attention(residual)
        
        return out


# In[ ]:


class UNet(nn.Module):
    def __init__(self, num_times, dim_hidden=128, mul_dim_list=[1, 2, 3, 4], attention_list=[2], num_resblocks=2, dropout=0., num_groups=32):
        super().__init__()
        assert all([i < len(mul_dim_list) for i in attention_list]), 'num_attention: index out of bounds.'
        
        dim_t = dim_hidden * 4
        self.time_embedding = TimeEmbedding(num_times, dim_hidden, dim_t)
        
        self.head = nn.Sequential(
            nn.Conv2d(3, dim_hidden, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, dim_hidden),
            nn.Mish(inplace=True),
        )
        
        self.downblocks = nn.ModuleList()
        downsample_list = []
        dim = dim_hidden
        for i, mul in enumerate(mul_dim_list):
            dim_out = dim_hidden * mul
            for _ in range(num_resblocks):
                self.downblocks.append(ResidualBlock(dim, dim_out, dim_t, dropout=dropout, attention=(i in attention_list)))
                dim = dim_out
            self.downblocks.append(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups, dim_out),
                nn.Mish(inplace=True),
            ))
            downsample_list += [dim]

        self.middleblocks = nn.Sequential(
            ResidualBlock(dim, dim, dim_t, dropout=dropout, attention=True),
            ResidualBlock(dim, dim, dim_t, dropout=dropout, attention=False),
        )

        self.upblocks = nn.ModuleList()
        for i, mul in reversed(list(enumerate(mul_dim_list))):
            dim_out = dim_hidden * mul
            dim_in = downsample_list.pop()
            modules = nn.ModuleList()
            modules.append(ResidualBlock(dim_in + dim, dim_out, dim_t, dropout=dropout, attention=(i in attention_list)))
            for _ in range(num_resblocks):
                modules.append(ResidualBlock(dim_out, dim_out, dim_t, dropout=dropout, attention=(i in attention_list)))
            self.upblocks.append(modules)
            self.upblocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups, dim_out),
                nn.Mish(inplace=True)
            ))
            dim = dim_out

        self.tail = nn.Conv2d(dim_out, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        t = self.time_embedding(t)
        
        x = self.head(x)
        
        xs = []
        for layer in self.downblocks:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
                xs.append(x)
        
        for layer in self.middleblocks:
            x = layer(x, t)

        for layer in self.upblocks:
            if isinstance(layer, nn.ModuleList):
                h = xs.pop()
                x = torch.cat([x, h], dim=1)
                for _layer in layer:
                    x = _layer(x, t)
            else:
                x = layer(x)
        
        x = self.tail(x)
        return x


# # Solver

# In[ ]:


class Solver:
    def __init__(self, args):
        has_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if has_cuda else "cpu")
        
        self.args = args
        self.epoch = 0
        
        self.model = UNet(self.args.num_times, dim_hidden=self.args.dim_hidden).to(self.device)
        self.model.apply(self.weights_init)

        self.diffusion_trainer = GaussianDiffusionTrainer(self.model, self.args.beta_1, self.args.beta_t, self.args.num_times).to(self.device)
        self.diffusion_sampler = GaussianDiffusionSampler(self.model, self.args.beta_1, self.args.beta_t, self.args.num_times).to(self.device)
        
        #self.optimizer = optim.Adam(self.model.parameters(), lr=2 * self.args.lr)
        self.optimizer = torch_optimizer.Lamb(self.model.parameters(), lr=2 * self.args.lr)
        self.scaler = GradScaler()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=4, eta_min=self.args.lr / 2)
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
            
    def load_dataset(self):
        self.dataloader = Util.loadImages(self.args.batch_size, self.args.image_dir, self.args.image_size)
        self.max_iters = len(iter(self.dataloader))
            
    def save_state(self, epoch):
        self.model.cpu()
        torch.save(self.model.state_dict(), os.path.join(self.args.weight_dir, f'weights.{epoch}.pth'))
        self.model.to(self.device)
        
    def load_state(self):
        if os.path.exists('weights.pth'):
            self.model.load_state_dict(torch.load('weights.pth', map_location=self.device))
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    @staticmethod
    def load(args, resume=True):
        if resume and os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                solver = load(f)
                print('Loaded resume.')
                return solver
        else:
            return Solver(args)
    
    def train(self, resume=True):
        self.load_dataset()
        
        print(f'Use Device: {self.device}')
        torch.backends.cudnn.benchmark = True
        
        self.model.train()
        
        hyper_params = {}
        hyper_params['Image Dir'] = self.args.image_dir
        hyper_params['Result Dir'] = self.args.result_dir
        hyper_params['Weight Dir'] = self.args.weight_dir
        hyper_params['Image Size'] = self.args.image_size
        hyper_params['Learning Rate'] = self.args.lr
        hyper_params["DDPM's beta_1"] = self.args.beta_1
        hyper_params["DDPM's beta_t"] = self.args.beta_t
        hyper_params["DDPM's Times"] = self.args.num_times
        hyper_params["UNet's dim_hidden"] = self.args.dim_hidden
        hyper_params['Batch Size'] = self.args.batch_size
        hyper_params['Num Train'] = self.args.num_train

        for key in hyper_params.keys():
            print(f'{key}: {hyper_params[key]}')
        #experiment.log_parameters(hyper_params)
        
        while self.args.num_train > self.epoch:
            self.epoch += 1
            epoch_loss = 0.0
            
            for iters, (data, _) in enumerate(tqdm(self.dataloader)):
                iters += 1
                losses = {}
                
                with torch.autocast(device_type='cuda' if str(self.device).startswith('cuda') else 'cpu'):
                    data = data.to(self.device, non_blocking=True)
                    loss = self.diffusion_trainer(data).sum() / self.args.num_times
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Logging.
                    losses['loss'] = loss.item()
                
                epoch_loss += losses['loss']
                #experiment.log_metrics(losses)

            self.save_state(self.epoch)
            print(f'Epoch[{self.epoch}]'
                  + f' LR[{self.scheduler.get_last_lr()[0]:.5f}]'
                  + f' Loss[{epoch_loss}]')

            ## DEBUG
            #self.generate()
            
            self.scheduler.step()
            
            if resume:
                self.save_resume()
    
    def generate(self):
        self.model.eval()
        noise = torch.randn(size=[1, 3, self.args.image_size, self.args.image_size], device=self.device)
        image = self.diffusion_sampler(noise)
        save_image(image, os.path.join(self.args.result_dir, f'generated_{time.time()}.png'))
        print('New picture was generated.')
        self.model.train()


# # Main

# In[ ]:


def main(args):
    solver = Solver.load(args, resume=not args.noresume)
    solver.load_state()
    
    if args.generate:
        solver.generate()
        return
    
    solver.train(not args.noresume)
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta_1', type=float, default=0.0001)
    parser.add_argument('--beta_t', type=float, default=0.02)
    parser.add_argument('--num_times', type=int, default=1000)
    parser.add_argument('--dim_hidden', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', action='store_true')
    #parser.add_argument('--noresume', action='store_true')
    
    args, unknown = parser.parse_known_args()
    args.noresume = True # Because, can not use for LAMB.

    if len(unknown) != 0:
        print(f'Unknown option: {unknown}')
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

