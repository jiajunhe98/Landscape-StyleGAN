# StyleGAN for Landscape Photographs



### **⭐Generate your own landscape photos [here](https://taking-non-existing-photos.herokuapp.com/)!⭐**

![image](https://github.com/hejj16/Landscape-StyleGAN/blob/main/result.png)


## This repository contains codes which:
- re-implement StyleGAN in PyTorch;
- progressively train StyleGAN on landscape photos;
- classify photos in the disentangled latent space;
- display the results with Dash and deploy the model on Heroku platform.



## Details

The StyleGAN models can be found in [```models```](https://github.com/hejj16/Landscape-StyleGAN/tree/main/models) directory. 
Implentmented tricks and features:
- [x] Progressive Training
- [x] PixelNorm Layers
- [x] Truncation Layers
- [x] Style Mixing
- [x] Loss with R1 Penalty
- [x] Gradient Clipping

The models are then trained using [this notebook](https://github.com/hejj16/Landscape-StyleGAN/blob/main/notebooks/StyleGAN_20210114_R1penalty.ipynb). After training, generated photos are classified by their time(day/night/dawn&dust) and weather(aurora or not) in the disentangled latent space using small neuron network in [this notebook](https://github.com/hejj16/Landscape-StyleGAN/blob/main/notebooks/classification.ipynb).

The codes for the web app can be found in the [```StyleGAN-Webpage```](https://github.com/hejj16/Landscape-StyleGAN/tree/main/StyleGAN-Webpage) directory.



## References
The codes rely heavily on NVIDIA's StyleGAN TF implementation:
<br />Karras, Tero, Samuli Laine, and Timo Aila. "A Style-Based Generator Architecture for Generative Adversarial Networks." *arXiv preprint arXiv:1812.04948* (2018). https://arxiv.org/abs/1812.04948
<br />Official StyleGAN implementation: https://github.com/NVlabs/stylegan

The implementation of Equalized Learning Rate layers refers to the following repository:
<br />https://github.com/huangzh13/StyleGAN.pytorch

The web pages are inspired by the following tutorial:
<br />Dash tutorial (Chinese): https://blog.csdn.net/lly1122334/article/details/107056777






