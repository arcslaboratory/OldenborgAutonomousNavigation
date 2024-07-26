# OldenborgModel

Train and perform inference on Oldenborg datasets.

## Setup

Create (or update) an environment with the following packages:

~~~bash
conda activate ENVIRONMENT
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
mamba install fastai wandb
~~~

## Architectures

A deeply flawed method for choosing an architecture:

- See [Papers With Code benchmarks](https://paperswithcode.com/task/image-classification)
- Go to the [Hugging Face models listing](https://huggingface.co/models)
- Set the task (eg, "Image Classification")
- Sort by "Trending" and by "Most Downloads"
- Look at model sizes and accuracies
- Set the library (eg, "timm") and search for the model name

`timm` models for these experiments:

| Full Name                                  | Params | Nice Name         |
| ------------------------------------------ | -----: | ----------------- |     
| resnet18.a1_in1k                           |  11.7M | ResNet18          |
| mobilenetv4_conv_small.e2400_r224_in1k     |   3.8M | MobileNetV4       |
| efficientnet_b3.ra2_in1k                   |  12.2M | EfficientNet      |
| convnextv2_atto.fcmae                      |   3.4M | ConvNextV2Atto    |
| convnextv2_base.fcmae_ft_in22k_in1k        |  88.7M | ConvNextV2Base    |
| vit_base_patch16_224.augreg2_in21k_ft_in1k |  86.6M | ViT               |
