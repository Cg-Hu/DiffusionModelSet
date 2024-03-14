import argparse
import os
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
import torch.nn.functional as F
# 免得sgd not found
import sys
sys.path.append("/home/hcg/bysj/SDG_code")
from sdg.parser import create_argparser
from sdg.logging import init_logging, make_logging_dir
from sdg.distributed import master_only_print as print
from sdg.distributed import is_master, init_dist, get_world_size
from sdg.gpu_affinity import set_affinity
from sdg.logging import init_logging, make_logging_dir
from sdg.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from sdg.clip_guidance import CLIP_gd
from sdg.image_datasets import load_ref_data
from sdg.misc import set_random_seed
from sdg.guidance import image_loss, text_loss
from sdg.image_datasets import _list_image_files_recursively
from torchvision import utils
import math
import clip
import random


def main():
    time0 = time.time()
    args = create_argparser().parse_args()
    # 设置对应GPU的CPU亲和性, 提高性能（通用代码）
    set_affinity(args.local_rank)
    if args.randomized_seed:
        args.seed = random.randint(0, 10000)
    # 设置随机种子（这个也是通用代码）
    set_random_seed(args.seed, by_rank=True)
    # 多个GPU分布式训练
    if not args.single_gpu:
        init_dist(args.local_rank)

    tb_log = None
    # 设置日志的存放路径
    args.logdir = init_logging(args.exp_name, root_dir='results', timestamp=False)

    if is_master():
        tb_log = make_logging_dir(args.logdir, no_tb=True)

    print("creating model...")
    # 创建一个Unet和Diffusion
    # model_and_diffusion_defaults()返回的是一个字典，提取这个函数的形参属性名，然后对应在args中提取该属性的值
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to('cuda')
    model.eval()

    # 这是原本的clip
    clip_model, preprocess = clip.load('RN50x16', device='cuda')

    # 这边就是语言引导，text_weight就是论文里的 s factor
    if args.text_weight == 0:
        instructions = [""]
    else:
        with open(args.text_instruction_file, 'r') as f:
            instructions = f.readlines()
    instructions = [tmp.replace('\n', '') for tmp in instructions]

    # define image list
    if args.image_weight == 0:
        imgs = [None]
    else:
        imgs = _list_image_files_recursively(args.data_dir)
        imgs = sorted(imgs)
    
    # 这个是微调后的CLIP
    clip_ft = CLIP_gd(args)
    clip_ft.load_state_dict(th.load(args.clip_path, map_location='cpu'))
    clip_ft.eval()
    clip_ft = clip_ft.cuda()

    def cond_fn_sdg(x, t, y, **kwargs):
        assert y is not None
        with th.no_grad():
            if args.text_weight != 0:
                # 原始的clip的文本编码器
                # 8x77 -> 8x768
                text_features = clip_model.encode_text(y)
            if args.image_weight != 0:
                target_img_noised = diffusion.q_sample(kwargs['ref_img'], t, tscale1000=True)
                target_img_features = clip_ft.encode_image_list(target_img_noised, t)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            # 这边就是微调的图像编码器
            # image_features [[tensor], []...,[]] 6个tensor最后一个8x768
            image_features = clip_ft.encode_image_list(x_in, t)
            if args.text_weight != 0:
                # size(8)
                loss_text = text_loss(image_features, text_features, args)
            else:
                loss_text = 0
            if args.image_weight != 0:
                loss_img = image_loss(image_features, target_img_features, args)
            else:
                loss_img = 0
            # args.text_weight condition factor，越大越受影响
            total_guidance = loss_text * args.text_weight + loss_img * args.image_weight
            # 计算损失 total_guidance 对输入张量 x_in 的梯度
            return th.autograd.grad(total_guidance.sum(), x_in)[0]


    print("creating samples...")
    count = 0
    for img_cnt in range(len(imgs)):
        # image guidance
        if imgs[img_cnt] is not None:
            print("loading data...")
            model_kwargs = load_ref_data(args, imgs[img_cnt])
        else:
            model_kwargs = {}

        for ins_cnt in range(len(instructions)):
            # 读出每行的文本内容
            instruction = instructions[ins_cnt]    
            # 一个文本取 batch_size times，[ins, ins, ins ..., ins]
            # 处理好的text->idx
            # text.shape = (batch_size x 77)，现在的text就是一个个的idx索引数字
            text = clip.tokenize([instruction for cnt in range(args.batch_size)]).to('cuda')
            model_kwargs['y'] = text
            # {'y': text} 文本guidance
            model_kwargs = {k: v.to('cuda') for k, v in model_kwargs.items()}
            if args.image_weight == 0 and args.text_weight == 0:
                cond_fn = None
            else:
                cond_fn = cond_fn_sdg

            # AMP:提高训练速度并减少内存使用
            with th.cuda.amp.autocast(True):
                sample = diffusion.p_sample_loop(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    noise=None,
                    clip_denoised=args.clip_denoised, # true
                    model_kwargs=model_kwargs, # {'y': text}
                    cond_fn=cond_fn, # 
                    device='cuda',
                )


            for i in range(args.batch_size):
                if args.text_weight == 0:
                    out_folder = '%05d_%s' % (img_cnt, os.path.basename(imgs[img_cnt]).split('.')[0])
                elif args.image_weight == 0:
                    out_folder = '%05d_%s' % (ins_cnt, instructions[ins_cnt])
                else:
                    out_folder = '%05d_%05d_%s_%s' % (img_cnt, ins_cnt, os.path.basename(imgs[img_cnt]).split('.')[0], instructions[ins_cnt])

                out_path = os.path.join(args.logdir, out_folder,
                                        f"{str(count * args.batch_size + i).zfill(5)}.png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                utils.save_image(
                    sample[i].unsqueeze(0),
                    out_path,
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

            count += 1
            print(f"created {count * args.batch_size} samples")
            print(time.time() - time0)

    print("sampling complete")


if __name__ == "__main__":
    main()
