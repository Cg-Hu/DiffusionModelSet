"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # 因为MPI安装不了，在主要是学习论文，对这些东西以后需要用到了再看。
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(th.load(args.model_path, map_location='cpu'))
    # model.to(dist_util.dev())
    device = th.device("cuda:0")
    if th.cuda.is_available():
        model.to(th.device("cuda:0"))
    th.device("cpu")
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # 这个操作可能用于将输入数据的范围从原来的范围映射到0到255之间的范围 .clamp()用于确保张量中的每个元素都在0到255之间。它将小于0的值变为0，将大于255的值变为255。这是为了确保像素值不超出8位图像的合法范围。
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # 这两个是分布式运算最后进行汇总的
        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        gathered_samples = sample
        
        all_images.extend([sample.cpu().numpy()[None] for sample in gathered_samples])
        if args.class_cond:
            # gathered_labels = [
            #     th.zeros_like(classes) for _ in range(dist.get_world_size())
            # ]
            # dist.all_gather(gathered_labels, classes)
            # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            all_labels.extend([labels.cpu().numpy()[None] for labels in classes])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # arr: (16,64,64,3) (N,H,W,C)
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)
    # /tmp/openai-2024-02-06-00-33-22-770771/samples_16x64x64x3.npz
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
