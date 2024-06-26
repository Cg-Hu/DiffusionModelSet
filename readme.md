# bysj-remote
various dm model, learning it by the technique.

# model

## 1. ddpm
最开始的学习的扩散模型
## 2. iddpm
加速采样、引入可学习的方差以及条件学习的dm
## 3. sdg
将CLIP引入dm，采用预训练的unconditional dm配合在噪声图像微调的CLIP Image Encoder,
无需retrain dm，就可以达到text & image 来指导图像的生成。
## 4. ldm
潜在空间生成，减小训练计算资源。
本模型参考这个
## 5. CFG
隐式的条件dm图像生成，将无条件和有条件的DM模型一起训练再以不同的权重进行组合而得，具体见显式条件指导的贝叶斯推到。
## 6. guided-classifier
暂无代码，引入预训练的分类器，并得出了均值偏移计算公式，使得类条件的加入是简单偏移了均值。
4.sdg也采用了这个方法，只不过将类变为text & image

## 7. vq-gan
本模型同样参考了这个vq的思想，以及压缩自编码器的设计准则。

## 第一阶段设计如下
* Logo图像生成的整体算法流程图
![Alt text](docImage/image.png)
* 本仓库采用的方法模型图
![Alt text](docImage/image-1.png)
  * 其中加入ASPP多尺度模块以及空间通道注意力机制提高第一阶段模型的压缩与特征分布学习能力
* 自编码器结构
* ![Alt text](docImage/image-2.png)
* ![Alt text](docImage/image-3.png)
* 本仓库认为新的网络设计原则有着如下优势：
* （1）参考VQ-GAN，在自编码器末尾加上Patch Discriminator。该操作提高了模型的学习能力，且只需要另外加几个卷积层，并且该层不会再一开始就参与网络计算，而是待自编码器训练到一定程度时引入该模型，从而避免了自编码初期效果差导致Patch Discriminator得到的损失梯度过大影响整个模型的训练。
（2）解码器上采样时使用插值和转置卷积两种操作。线性插值和转置卷积组合上采样操作既保证了上采样有效地进行，也在一定程度降低了模型参数（插值层不需要参数）。
（3）引入了多尺度融合操作ASPP，并以两种尺度的特征进行学习训练。ASPP通过不同尺度的空洞卷积和金字塔池化，有效地捕获了输入图像中不同尺度的信息，使网络具备多尺度感知能力。并且空洞卷积通过调整卷积核的采样率，扩大了感受野，使网络能够更好地捕捉物体的全局信息。与此同时网络同时使用最后一层下采样和倒数第二层下采样得到的特征进行学习训练，这可使第二阶段DM模型生成两种尺度的数据。通过实验证明，本文的自编码器学习到的分布特征是一种更深层次的特征（与尺度无关），可以实现多种尺度特征的解码还原。
（4）引入了CBAM注意力机制。CBAM通过整合通道和空间注意力，能够对多尺度的特征进行建模。这有助于网络更好地理解物体的全局和局部特征，提高对不同尺度目标的识别性能。相较于自注意力机制，CBAM的通道和空间注意力模块相对轻量，计算效率更高。并且相对于自注意力机制的全局性，CBAM更注重局部性，通过通道和空间注意力模块来捕捉特定区域和通道的重要性。

## 第二阶段
CFG+DDIM(Unet)
1. 采用Transformer结构替代Unet(ing)
2. 采用CLIP结构替代纯Transformer结构的文字特征提取头(√)
3. 对生成的Logo采用“图像中心部分提取算法“，避免无关的部分对Logo造成影响。(√)

* 如下图所示，为第二阶段U-Net的网络架构(ResBlock Attention with cross and cbam)
  * ![Alt text](docImage/Unet.png)
* 图像中心提取算法
  * 取自纯OpenCV的方法（数字图像处理）
  * <img src="docImage/extract.jpg" style="zoom:40%;" />

* 图像编辑（待做）
  * AnyText论文
  * 代码跑通
  * 复现估计整不了，看看能不能微调，至少不能比直接使用它API差。
  *  <img src="docImage/anytext.jpg" style="zoom:30%;" />

