import torch
import numpy as np


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0) # 限定方差在[-30,20]之间
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar) # 标准差
        self.var = torch.exp(self.logvar) # 方差
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x
    # KL散度： 通常比较的是模型学习到的潜在变量的分布（通常假设为高斯分布，由mean和logvar参数化）与标准正态分布之间的差异。
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

'''
其中那个Parameter就是encode x 之后的输出
这个类DiagonalGaussianDistribution定义了一个具有对角协方差矩阵的高斯（正态）分布。这种分布通常用于机器学习和深度学习中，特别是在变分自编码器（VAEs）和其他生成模型中。下面是对该类及其方法的解释：
__init__ 方法
初始化方法接收两个参数：parameters和deterministic。
parameters应该是一个张量，其中包含分布的均值和对数方差，这些参数被一同传入，并通过torch.chunk在第二维上被平均分成两部分，分别代表均值（mean）和对数方差（logvar）。
logvar被限制在-30到20之间，以避免数值稳定性问题。
如果deterministic为True，则表示分布应该是确定性的，此时标准差和方差都设置为与均值形状相同且全为零的张量。这通常用于测试或推理阶段，以确保输出是确定的。
sample 方法
生成一个采样，根据均值和标准差生成正态分布的随机样本。如果deterministic为True，则每次返回的样本将与均值相同。
kl 方法
计算当前分布与另一个高斯分布之间的Kullback-Leibler散度（KL散度）。KL散度是衡量两个概率分布差异的指标。
如果没有提供另一个分布（即other为None），则假设另一个分布是标准正态分布（均值为0，方差为1）。
如果deterministic为True，则返回0，因为确定性分布之间的KL散度没有意义。
nll 方法
计算给定样本的负对数似然（Negative Log-Likelihood, NLL）。这是评估样本如何适应当前分布的一种方法。
如果deterministic为True，则返回0，因为确定性分布的NLL没有意义。
mode 方法
返回分布的模式（最常见的值），对于高斯分布，其模式与均值相同。
这个类提供了处理和分析对角高斯分布的基本工具，特别适用于需要参数化概率分布的深度学习应用，如变分自编码器。通过这种方式，模型可以学习生成数据的分布，并能够从该分布中采样。
'''
