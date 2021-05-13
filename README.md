# StyleGAN for Landscape Photographs

## ***......Being Updated......***


**⭐Generate your own landscape photos [here](https://taking-non-existing-photos.herokuapp.com/)!⭐**

![image](https://github.com/hejj16/Landscape-StyleGAN/blob/main/result.png)
 <br />Uncurated set of images producted by the generator. 

### This repository contains codes which:
- re-implement StyleGAN in PyTorch;
- progressively train StyleGAN on landscape photos;
- classify photos by time and weather in the disentangled latent space;
- display the results with Dash and deploy the model on Heroku platform.


### Details:

The StyleGAN models can be found in ```models``` directory. 
Implentmented tricks and features:
- [x] Progressive Training
- [x] PixelNorm Layers
- [x] Truncation Layers
- [x] Style Mixing
- [x] Loss with R1 Penalty
- [x] Gradient Clipping

The models are then trained using [this notebook](https://github.com/hejj16/Landscape-StyleGAN/blob/main/notebooks/StyleGAN_20210114_R1penalty.ipynb) 






