import numpy as np
import math

import torch
import torch.nn.functional as F

def gram_matrix(input):
    if input.dtype == torch.float16:
        input = input.to(torch.float32)
        flag = True
    a, b, c, d = input.size()  # a=batch size(=1)
    sqrt_sum = math.sqrt(a * b * c * d)  # for numerical stability
    features = input.view(a * b, c * d) / sqrt_sum  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    result = G
    if flag:
        return result.to(torch.float16)
    else:
        return result

def image_loss(source, target, args):
    if args.image_loss == 'semantic':
        source[-1] = source[-1] / source[-1].norm(dim=-1, keepdim=True)
        target[-1] = target[-1] / target[-1].norm(dim=-1, keepdim=True)
        return (source[-1] * target[-1]).sum(1)
    elif args.image_loss == 'style':
        weights = [1, 1, 1, 1, 1]
        loss = 0
        for cnt in range(5):
            loss += F.mse_loss(gram_matrix(source[cnt]), gram_matrix(target[cnt]))
        return -loss * 1e10 / sum(weights)

# image_feature 只取最后一个
def text_loss(source, target, args):
    # 每一行，也就是每一个样本算L2范数 = sqrt(x1^2 + x2^2 + ... + xn^2)
    # 然后每一行除以算出来这行的L2范数进行归一化
    source_feat = source[-1] / source[-1].norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)
    # 计算内积，对应位置相乘，这个内积表示相似性，因为单位长度的向量之间的点积等于它们之间的余弦相似度
    # 这就表示source样本1与target样本2的相似度
    # 一号图像样本与text guidacne condition的相似度
    return (source_feat * target).sum(1)
