# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools
import ctypes

import torch
import torch.distributed as dist


def init_dist(local_rank, backend='nccl', **kwargs):
    r"""Initialize distributed training"""
    if dist.is_available():
        if dist.is_initialized():
            return torch.cuda.current_device()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method='env://', **kwargs)

    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL('libcudart.so')
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    # assert pValue.contents.value == 128


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    # 这些都是深度学习pytorch框架分布式训练的工作
    if dist.is_available():
        if dist.is_initialized():
            # dist.get_rank() 就是用于获取当前进程的排名，以便进行相应的分布式操作
            rank = dist.get_rank() # 返回就是 0,1,2,3 (四个卡的主机下)
    return rank


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def master_only(func):
    r"""Apply this function only to the master GPU."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r"""Simple function wrapper for the master function"""
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


def is_local_master():
    return torch.cuda.current_device() == 0


@master_only
def master_only_print(*args, **kwargs):
    r"""master-only print"""
    print(*args, **kwargs)

def print_all_rank(*args, **kwargs):
    rank = get_rank()
    print('[rank %d] %s' % (rank, *args), **kwargs)


def dist_reduce_tensor(tensor, rank=0, reduce='mean'):
    r""" Reduce to rank 0 """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=rank)
        if get_rank() == rank:
            if reduce == 'mean':
                tensor /= world_size
            elif reduce == 'sum':
                pass
            else:
                raise NotImplementedError
    return tensor


def dist_all_reduce_tensor(tensor, reduce='mean'):
    r""" Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        if reduce == 'mean':
            tensor /= world_size
        elif reduce == 'sum':
            pass
        else:
            raise NotImplementedError
    return tensor


def dist_all_gather_tensor(tensor):
    r""" gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    tensor_list = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)
    return tensor_list

class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype
        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)


def all_gather_with_gradient(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    return AllGatherFunction.apply(tensor)
