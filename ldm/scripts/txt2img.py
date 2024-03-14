import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
# 免得ldm not found
sys.path.append("/home/hcg/bysj-remote/ldm")
# sys.path.append("/home/hcg/bysj-remote/logo-DM")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import time


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.to(torch.device("cuda:2"))
    model.eval()
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        # 这个就是CFG的思想，不过这玩意的形式和CFG推到出来的不一样
        # CFG是长这样 eps(x, c) + w * (eps(x, cond) - eps(x, empty))"
    )
    opt = parser.parse_args()


    config = OmegaConf.load("/home/hcg/bysj-remote/ldm/configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    # model = load_model_from_config(config, "/home/hcg/bysj-remote/ldm/models/ldm/text2img-large/model.ckpt")  # TODO: check path
    # 得到整个的LatentDiffusionModel并且加载好参数（里面包括AutoEncoderKL，first stage的编码解码器）
    model = load_model_from_config(config, "/home/hcg/bysj_param_dataset/ldm/models/ldm/text2img-large/model.ckpt")  # TODO: check path

    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    start_time = time.time()
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope(): # 初始是没有使用EMA
            uc = None # uc 和 c的 编码方式是一样的 (4,77,1280) (N,seqlen,dim)
            if opt.scale != 1.0: # 如果这玩意等于1.0，就相当于完全没有Unconditional，
               uc = model.get_learned_conditioning(opt.n_samples * [""]) # cfg中说过，无条件就是用0.文本的话显然就是空, # 得到经过SelfAttention的条件编码
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt]) # 纯Transformer做的文字特征提取
                shape = [4, opt.H//8, opt.W//8] # 缩小了8倍，但是这边的通道数是4
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)
                # 现在得到的图像为 4,4,32,32(原图256)，现在要从Latent Space解码回去
                x_samples_ddim = model.decode_first_stage(samples_ddim) # decode得到原图大小，最终得到了这一步的图像
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0) # 映射到0,1之间

                for x_sample in x_samples_ddim: # batch_size
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c') # 变到0带255之间
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png")) # 这行代码是使用 PIL 库中的 Image.fromarray() 方法将一个 NumPy 数组 x_sample 转换为图像，并保存为 PNG 格式的文件。
                    base_count += 1
                all_samples.append(x_samples_ddim)

    end_time = time.time()
    print (f"Logo generation time is:{end_time - start_time}_s")
    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    # print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
    print(f"Your Logos or trademarks have be generated!")
