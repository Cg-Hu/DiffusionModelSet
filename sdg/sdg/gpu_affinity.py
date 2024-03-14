# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import math
import os
import pynvml

pynvml.nvmlInit()


def systemGetDriverVersion():
    r"""Get Driver Version"""
    return pynvml.nvmlSystemGetDriverVersion()


def deviceGetCount():
    r"""Get number of devices"""
    return pynvml.nvmlDeviceGetCount()


class device(object):
    r"""Device used for nvml."""
    # 这行代码的作用是计算将CPU核心数（CPU cores）分组成NVIDIA GPU亲和性（affinity）元素所需的数量。
    # 以便在GPU编程或任务调度中更有效地管理CPU和GPU之间的资源分配。这个值通常用于优化多线程或多进程应用程序，以充分利用系统的硬件资源。
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)

    def __init__(self, device_idx):
        super().__init__()
        ''''
        pynvml 是一个Python库，用于与NVIDIA的管理库NVML（NVIDIA Management Library）交互，以便查询和控制GPU的状态和性能。
        nvmlDeviceGetHandleByIndex(device_idx) 是pynvml库中的一个函数。它接受一个参数 device_idx，表示要操作的GPU的索引。
        这个函数的作用是根据传入的索引 device_idx 获取对应GPU的句柄（handle），这个句柄可以用来后续查询和设置GPU的各种属性和状态。
        通常，这种操作用于编写需要与GPU进行交互的程序，例如获取GPU的温度、显存使用情况、性能信息等。要使用这个函数，需要安装pynvml库并且在计算机上有NVIDIA GPU。
        '''
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        # print(f"0号GPU句柄为: {self.handle}")

    def getName(self):
        r"""Get obect name"""
        # 使用句柄来获取对应GPU的名称或型号。它会返回一个包含GPU名称的字符串，例如 "NVIDIA GeForce GTX 1080" 或 "NVIDIA Tesla V100".
        return pynvml.nvmlDeviceGetName(self.handle)

    def getCpuAffinity(self):
        r"""Get CPU affinity"""
        affinity_string = ''
        for j in pynvml.nvmlDeviceGetCpuAffinity(
                self.handle, device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            # print(f"64位整数: {j}")
            # print('{:064b}'.format(j))
            affinity_string = '{:064b}'.format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list
        # print(affinity_list)
        # 返回一个只包含非零元素的索引列表，这些非零元素表示GPU_dev_id与哪些CPU核心存在亲和性
        # 这对于在GPU编程或任务调度中确定哪些CPU核心与特定GPU相关联是很有用的。
        return [i for i, e in enumerate(affinity_list) if e != 0]


def set_affinity(gpu_id=None):
    r"""Set GPU affinity

    Args:
        gpu_id (int): Which gpu device.
    """
    # 如果GPU不在则默认使用0号GPU
    if gpu_id is None:
        gpu_id = int(os.getenv('LOCAL_RANK', 0))

    dev = device(gpu_id)
    # 当前进程的CPU亲和性将被设置为与特定GPU相关联的CPU核心
    # 这种操作通常用于GPU编程或任务调度中，以确保计算任务在特定CPU核心上执行，以最大程度地提高性能并避免资源竞争。
    # 这种设置可以确保GPU与CPU之间的数据传输和计算任务在适当的CPU核心上进行，以提高整体性能。
    os.sched_setaffinity(0, dev.getCpuAffinity())

    # list of ints
    # representing the logical cores this process is now affinitied with
    return os.sched_getaffinity(0)

if __name__ == '__main__':
    dev = device(0)
    print(f'gpu的名字为: {dev.getName()}')
    dev.getCpuAffinity()






