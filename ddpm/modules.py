
import torch
import torch.nn as nn
import torch.nn.functional as F


# 就按照一个公式，简单粗暴的对模型的参数进行遍历更改
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    # ema 不是一开始就有
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

# 大小和通道都不变
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        # 这种模块通常被用在自注意力机制中的前馈神经网络（Feed-Forward Neural Network）中，以提高模型的表现
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(), # GELU是一种激活函数，，增强网络的非线性
            nn.Linear(channels, channels),
        )

    def forward(self, x, img_size):
        # sa1 x: torch.size([bs,128,256*256])
        # batchsize x seqlen x embedding
        # 在这边看来，一个图像的256*256就是一个样本，embeding是通道数
        #  基于通道(特征维度)做的注意力机制
        # x -> torch.size([bs,256*256, 128])
        x = x.view(-1, self.channels, img_size * img_size).swapaxes(1, 2) # 交换1和2两个维度
        x_ln = self.ln(x) # 对最后一个维度channels作归一化
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) # 多头注意力机制
        attention_value = attention_value + x  # 残差连接
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, img_size, img_size)


# 大小不变
# channels: 3->64
# 无residual
# size: 512->512 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

# down1
# channels: 64->128
# size: 512->256 
# down2
# channels: 128->256
# size: 256->128         
# down3
# channels: 256->256
# size: 128->64          
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 大小变一半
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            # 通道的编码，加在每个通道的Feature Map上
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        # x: torch.size([1, 128, 256, 256])
        x = self.maxpool_conv(x)
        # emb:torch.size([1,128])
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb # broadcasting


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        # scale_factor指定输出尺寸是输入尺寸的两倍，双线性插值
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    # U-net中的跳跃连接 cat
    def forward(self, x, skip_x, t):
        x = self.up(x)
        # print(f'上采样之后的大小:{x.shape}')
        # 换成512x512这个维度出现了点问题torch.Size([64, 256, 16, 16]) 已解决，需要把SA中的size不能直接写死

        # 这个均值的代码怎么突然会在这？
        # x = torch.mean(x, dim=0)
        # print(f'x进行batch维度求和之后:{x.shape}')
        # skip_x = torch.mean(skip_x, dim=0).unsqueeze(0)
        # print(f'skip_x进行batch维度求和之后:{skip_x.shape}').unsqueeze(0)

        x = torch.cat([skip_x, x], dim=1) # torch.Size([1, 512, 8, 8])
        # print(f'cat:{x.shape}')
        x = self.conv(x) # torch.Size([1, 128, 8, 8])
        # print(f'conv:{x.shape}')
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # torch.Size([1, 128, 1, 1])
        # torch.Size([1, 128, 8, 8])
        return x + emb # 数据+通道位置编码


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)              
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        # inv_freq: torch.size([128])
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        # 这边实际算的是点积 这个看做是一个向量inv_freq
        # [batch_size x channels // 2] * [channels // 2]
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc

    def forward(self, x, t):                    # 输入x 为 4x3x64x64
        t = t.unsqueeze(-1).type(torch.float)   # t: torch.size([bs, 1]) 
        # self.time_dim = 128
        t = self.pos_encoding(t, self.time_dim) # t: torch.size([bs, 128])  
                                                # x: # torch.Size([1, 3, 64, 64])

        x1 = self.inc(x)                        # torch.Size([1, 64, 64, 64])
        x2 = self.down1(x1, t)                  # torch.Size([1, 128, 32, 32])
        # print(f'x2.shape is : {x2.shape}')      
        x2 = self.sa1(x2, x2.shape[-1])         # torch.Size([1, 128, 32, 32])
        # print(f'x2.shape is : {x2.shape}')
        x3 = self.down2(x2, t)                  # torch.Size([1, 256, 16, 16])
        x3 = self.sa2(x3, x3.shape[-1])         # torch.Size([1, 256, 16, 16])
        # print(f'x3.shape is : {x3.shape}')
        x4 = self.down3(x3, t)                  # torch.Size([1, 256, 8, 8])
        x4 = self.sa3(x4, x4.shape[-1])         # torch.Size([1, 256, 8, 8])
        # print(f'x4.shape is : {x4.shape}')
        
        x4 = self.bot1(x4)                      # torch.Size([1, 512, 8, 8])
        x4 = self.bot2(x4)                      # torch.Size([1, 512, 8, 8])
        x4 = self.bot3(x4)                      # torch.Size([1, 256, 8, 8])
        # print(f'x4.shape is : {x4.shape}')

        # 上采样开始 跳跃连接
        x = self.up1(x4, x3, t)                 # torch.Size([1, 128, 16, 16])
        # print(f'x.shape is : {x.shape}')

        x = self.sa4(x, x.shape[-1])            # torch.Size([1, 128, 16, 16])
        x = self.up2(x, x2, t)                  # torch.Size([1, 64, 32, 32])
        x = self.sa5(x, x.shape[-1])            # torch.Size([1, 64, 32, 32])
        x = self.up3(x, x1, t)                  # torch.Size([1, 64, 64, 64])
        x = self.sa6(x, x.shape[-1])            # torch.Size([1, 64, 64, 64])
        output = self.outc(x)                   # torch.Size([1, 3, 64, 64])
        # print(f'output shape is: {output.shape}')
        return output

class UNet_512(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=8), # 大小变一半
        )
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)              
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        # torch.size([128])
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        # 这边实际算的是点积 这个看做是一个向量inv_freq
        # [batch_size x channels // 2] * [channels // 2]
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float) 
        t = self.pos_encoding(t, self.time_dim) # t: torch.size([bs, 128])
                                                # x: # torch.Size([1, 3, 64, 64])
        x = self.maxpool_conv(x)
        # x = self.maxpool_conv(x)
        # x = self.maxpool_conv(x)
        x1 = self.inc(x)                        # torch.Size([1, 64, 64, 64])
        x2 = self.down1(x1, t)                  # torch.Size([1, 128, 32, 32])
        # print(f'x2.shape is : {x2.shape}')      
        x2 = self.sa1(x2, x2.shape[-1])         # torch.Size([1, 128, 32, 32])
        # print(f'x2.shape is : {x2.shape}')
        x3 = self.down2(x2, t)                  # torch.Size([1, 256, 16, 16])
        x3 = self.sa2(x3, x3.shape[-1])         # torch.Size([1, 256, 16, 16])
        # print(f'x3.shape is : {x3.shape}')
        x4 = self.down3(x3, t)                  # torch.Size([1, 256, 8, 8])
        x4 = self.sa3(x4, x4.shape[-1])         # torch.Size([1, 256, 8, 8])
        # print(f'x4.shape is : {x4.shape}')
        
        x4 = self.bot1(x4)                      # torch.Size([1, 512, 8, 8])
        x4 = self.bot2(x4)                      # torch.Size([1, 512, 8, 8])
        x4 = self.bot3(x4)                      # torch.Size([1, 256, 8, 8])
        # print(f'x4.shape is : {x4.shape}')


        x = self.up1(x4, x3, t)                 # torch.Size([1, 128, 16, 16])
        # print(f'x.shape is : {x.shape}')

        x = self.sa4(x, x.shape[-1])            # torch.Size([1, 128, 16, 16])
        x = self.up2(x, x2, t)                  # torch.Size([1, 64, 32, 32])
        x = self.sa5(x, x.shape[-1])            # torch.Size([1, 64, 32, 32])
        x = self.up3(x, x1, t)                  # torch.Size([1, 64, 64, 64])
        x = self.sa6(x, x.shape[-1])            # torch.Size([1, 64, 64, 64])
        
        x = self.outc(x)                   # torch.Size([1, 3, 64, 64])
        output = self.up_8(x)
        # print(f'output shape is: {output.shape}')
        # exit()
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
