import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import socket
import yaml

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_cifar100(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    mse_num = 1e9
    for epoch in range(args.epochs):
        loss_one_epoch = 0
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            # CFG的思想，无条件和有条件的联合训练
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            loss_one_epoch += loss.item()
        
        if loss_one_epoch < mse_num:
            print("更新了一次最小的MSE")
            mse_num = loss_one_epoch
            torch.save(model.state_dict(), os.path.join(args.log_name, "model_paras", "best_ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.log_name, "model_paras", f"best_ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.log_name, "model_paras", f"best_optim.pt"))
       


        
        labels = torch.arange(10).long().to(device)
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
        save_images(sampled_images, os.path.join(args.log_name, "img_results", f"{epoch}.jpg"))
        save_images(ema_sampled_images, os.path.join(args.log_name, "img_results", f"{epoch}_ema.jpg"))
        torch.save(model.state_dict(), os.path.join(args.log_name, "model_paras", f"last_ckpt.pt"))
        torch.save(ema_model.state_dict(), os.path.join(args.log_name, "model_paras", f"last_ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join(args.log_name, "model_paras", f"last_optim.pt"))
    

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.record_dir = "record_dir"
    # os.makedirs(args.record_dir)
    args.run_name = "DDPM_conditional_cifar10"
    args.log_name = f"/home/hcg/bysj-remote/ddpm/{args.record_dir}/{args.run_name}"
    os.makedirs(args.log_name, exist_ok=True)
    os.makedirs(os.path.join(args.log_name, f"{datetime.now().strftime('%b%d_%H:%M:%S')}-{socket.gethostname()}"), exist_ok=True)

    
    args.epochs = 300
    args.batch_size = 4
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = "/home/hcg/bysj-param-dataset/dataset/cifar10/train" # 采用CIFAR10数据集
    args.device = "cuda:0"
    args.lr = 3e-4
    seed = 314
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    with open(os.path.join(args.log_name, 'config.yml'), 'w+') as file:
        vars(args)
        yaml.dump(args, file)
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

