# DDPM for EIT

## 生成式模型

- 生成式模型学习数据的概率分布，以便我们可以从该分布中采样出这类数据。比如，通过从一组包含各式各样的猫的图片中学习这类数据的概率分布，我们可以从这个分布中获得一张新的关于猫的图片。


## 卷积神经网络

- convolution visualizer

- 我们知道卷积神经网络在图像分割以及有关计算机视觉的任务中表现非常优秀

## Encoder
- 减小了图像的尺寸同时增加了图像的特征通道(通道的数量)， 因此图像中的像素点数量变小了但是每一个像素所表示的信息量在多个通道上增加了。

- 学习了一个latent space 是一个多变量的高斯分布。
- 本质上训练encoder用来学习miu & sigma -> 均值和方差 分布， 不是直接压缩图像

- scale 
    - https://github.com/huggingface/diffusers/issues/437


## pic
 - attention 1:33:59