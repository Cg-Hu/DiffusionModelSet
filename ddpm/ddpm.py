import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, UNet_512
import logging
from torchvision.utils import save_image
import socket
from datetime import datetime
import yaml
import cv2
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# 默认步数是1000步
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda:1"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # self.img_size = img_size
        self.img_size = 128
        self.device = device

        # 这边写了在cuda上，后面依赖这个数据而生的数据都在cuda上
        # 这就是公式上的两个参数
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        # 累积乘法
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    # 线性划分
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        x = x.to(self.device) # 这边必须要赋值回去
        t = t.to(self.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(self.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(self.device)
        Ɛ = torch.randn_like(x).to(self.device)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    # 生成一个整数[1, high)
    # torch.Size([1])
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    # model是Attention Unet, n是sample num
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 生成n张随机噪声，作为xt时间步结果
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # [999, 998, ..., 1]
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # n张图片都需要加上时间步t
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None] # 维度扩展
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # 这边在区分是否到达x1
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # 推导出来的公式（得到噪声计算均值和方差，通过重采样的带下一个时间的图像）
                # 4x3x128x128 4x1x1x1 广播计算最终算出来的均值大小和x一致，
                # 这边直接把 beta当做方差了，重参数化技巧
                # torch.sqrt(beta) * noise 重参数化技巧，每一步都会引入随机性
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.log_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(device=args.device).to(device)
    # model = UNet_512().to(device)
    # ckpt = torch.load("./models/DDPM_Uncondtional/ckpt.pt")
    # model.load_state_dict(ckpt)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join(args.log_name, "runs"))
    l = len(dataloader)
    mse_num = 1e9
    loss_epoch = 0
    batch_num = 0
    for epoch in range(args.epochs):
        loss_epoch = 0
        batch_num = 0
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, ncols=95)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            # t的大小是batchsize，得到noise_steps其中的一步赋给t
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # x_t就是按照公式计算得到的噪声图像
            x_t, noise = diffusion.noise_images(images, t)
            # print(x_t.shape)
            # print(t.shape)
            # exit()
            # 明确一个点：加噪声环节是一次性计算得到的，但是去噪是要按照一步一步来
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            loss_epoch += loss.item()
            batch_num += 1
        loss_epoch_avg = loss_epoch / (batch_num + 1e-9)
        sampled_images = diffusion.sample(model, n=2)
        save_images(sampled_images, os.path.join(args.log_name, "img_results", f"{epoch}.jpg"))
        if loss_epoch_avg < mse_num:
            print("更新了一次最小的MSE")
            mse_num = loss_epoch_avg
            torch.save(model.state_dict(), os.path.join(args.log_name, "model_paras", "best_mse_ckpt.pt"))
        torch.save(model.state_dict(), os.path.join(args.log_name, "model_paras", "last_ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.record_dir = "record_dir"
    # os.makedirs(args.record_dir)
    args.run_name = f"DDPM_Uncondtional_Unet"
    args.log_name = f"{args.record_dir}/{args.run_name}/{datetime.now().strftime('%b%d_%H:%M:%S')}-{socket.gethostname()}"
    os.makedirs(args.log_name)
    args.epochs = 500
    args.batch_size = 2
    # args.image_size = 64
    # args.dataset_path = "../dataset/64x64/train"
    args.image_size = 512
    args.dataset_path = "../dataset/512x512/train"
    args.device = "cuda:0"
    args.lr = 3e-4
    seed = 314
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
    with open(os.path.join(args.log_name, 'config.yml'), 'w+') as file:
        vars(args)
        yaml.dump(args, file)
    train(args)

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    if not os.path.exists(filename):
        os.mkdir(filename)
    assert (len(input_tensor.shape) == 4)
    save_image(input_tensor, os.path.join(filename, f'{2}.jpg'))
    # for idx, img in enumerate(input_tensor):
    # 复制一份
        
        # print(f'img type is: {type(img)}')
        # print(f'img size is: {img.shape}')
        # img = img.clone().detach()
        # # 到cpu
        # img = img.to(torch.device('cpu'))
        # # 反归一化
        # # input_tensor = unnormalize(input_tensor)
        # # 去掉批次维度
        # # input_tensor = input_tensor.squeeze()
        # # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
        # img = img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        # # RGB转BRG
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(filename, f'{idx}.jpg'), img)


if __name__ == '__main__':
    # train
    launch()

    # sample
    # device = "cuda:2"
    # model = UNet(device=device).to(device)
    # print("loading checkpoints")
    # ckpt = torch.load("/home/hcg/bysj/DM/best_mse_ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=512, device=device)
    # x = diffusion.sample(model, 4)
    # save_images(x, f"./{5}.jpg")
    
    # x = x.permute(1, 2, 0)
    # x = x.squeeze(0)
    # x = x.cpu().numpy()
    # print(x.shape)
    # print(f'图片的类型为: {type(x)}')
    # x = x.transpose(1, 2, 0)
    # cv2.imwrite(f"/home/hcg/bysj/Diffusion-Models-pytorch/record_dir/DDPM_Uncondtional_Unet/Jan06_14:28:41-VIPA207/sample/{1}.jpg", x)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
    # save_image_tensor2cv2(x, "./record_dir/DDPM_Uncondtional_Unet512/Jan05_16:39:22-ubuntu/sample_results")