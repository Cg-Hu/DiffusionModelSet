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
潜在空间生成，减小训练计算资源
## 5. CFG
暂无代码，隐式的条件dm图像生成
## 6. guided-classifier
暂无代码，引入预训练的分类器，并得出了均值偏移计算公式，使得类条件的加入是简单偏移了均值。
4.sdg也采用了这个方法，只不过将类变为text & image

