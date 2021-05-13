import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

z_dim = 512
w_dim = 512
mapping_layers = 8
mapping_activation = "LeakyReLU"
device = "cuda" if torch.cuda.is_available() else "cpu"




#########################
## Self-defined Layers ##
#########################


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


class StddevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features,
                       c // self.num_new_features, h, w])
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = (y + 1e-8) ** 0.5
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, self.num_new_features, h, w)
        z = torch.cat([x, y], dim=1)
        return z


class Upscale2d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        factor = 2
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x


class Downscale2d(nn.Module):

    def __init__(self):
        super().__init__()
        factor = 2
        f = [1 / factor] * factor
        self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)

    def forward(self, x):
        return self.blur(x)


class EqualizedLinear(nn.Module):

    def __init__(self, input_size, output_size, gain=2 ** 0.5, lrmul=0.01):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.

        init_std = 1.0 / lrmul
        self.w_mul = he_std * lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        self.bias = torch.nn.Parameter(torch.zeros(output_size))
        self.b_mul = lrmul

    def forward(self, x):
        bias = self.bias
        bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class EqualizedConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** 0.5,
                 lrmul=1, upscale=False, downscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        init_std = 1.0 / lrmul
        self.w_mul = he_std * lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)

        self.bias = torch.nn.Parameter(torch.zeros(output_channels))
        self.b_mul = lrmul

        self.intermediate = None

    def forward(self, x):
        bias = self.bias
        bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            w = F.pad(w, [1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, [1, 1, 1, 1])
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
            x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class Truncation(nn.Module):

    def __init__(self, avg_latent, max_layer=9, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x, threshold):
        interp = torch.lerp(self.avg_latent, x, threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer * 2).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)


###########################
## Noise Mapping Network ##
###########################


class MappingNet(nn.Module):
    """
    Noise Mapping Network. Map latent Z to disentangled latent W.

    Args:
        z_dim: Default 512
        w_dim: Default 512
        h_dim: Hidden dim, default 512
        mapping_layers: Default 8
        mapping_activation: {"LeakyReLU", "ReLU"}, Default "LeakyReLU"
        w_broadcast: If not None, the output size will be (batch_size, w_broadcast, w_dim)

    """

    def __init__(self,
                 z_dim=512,
                 w_dim=512,
                 h_dim=512,
                 mapping_layers=8,
                 mapping_activation="LeakyReLU",
                 w_broadcast=18
                 ):
        super().__init__()
        self.w_broadcast = w_broadcast
        activation = nn.ReLU() if mapping_activation == "ReLU" else nn.LeakyReLU(0.2)
        self.mappingnet = nn.Sequential(PixelNormLayer())

        for layer in range(mapping_layers):
            if layer == 0:
                self.mappingnet.add_module("linear" + str(layer), EqualizedLinear(z_dim, h_dim))
                self.mappingnet.add_module("act" + str(layer), activation)
            elif layer == mapping_layers - 1:
                self.mappingnet.add_module("linear" + str(layer), EqualizedLinear(h_dim, w_dim))
                self.mappingnet.add_module("act" + str(layer), activation)
            else:
                self.mappingnet.add_module("linear" + str(layer), EqualizedLinear(h_dim, h_dim))
                self.mappingnet.add_module("act" + str(layer), activation)

    def forward(self, x):

        x = self.mappingnet(x)

        if self.w_broadcast:
            x = x.unsqueeze(1).expand(-1, self.w_broadcast, -1)
        return x


#######################
## Synthesis Network ##
#######################


class SynthesisBlock(nn.Module):
    """
    One Block in Synthesis Net.

    Overall structure:
        Conv&Upsample --> Add Noise --> Activation --> AdaIN --> Conv --> Add Noise --> Activation --> AdaIN

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_size=w_dim,
                 ):
        super(SynthesisBlock, self).__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, upscale=True)
        self.noise_weight1 = nn.Parameter(
            torch.randn((1, out_channels, 1, 1))
        )
        self.activation1 = nn.LeakyReLU(0.2)
        self.instance_norm1 = nn.InstanceNorm2d(out_channels)
        self.style_scale_transform1 = EqualizedLinear(latent_size, out_channels, gain=1.0)
        self.style_shift_transform1 = EqualizedLinear(latent_size, out_channels, gain=1.0)

        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3)
        self.noise_weight2 = nn.Parameter(
            torch.randn((1, out_channels, 1, 1))
        )
        self.activation2 = nn.LeakyReLU(0.2)
        self.instance_norm2 = nn.InstanceNorm2d(out_channels)
        self.style_scale_transform2 = EqualizedLinear(latent_size, out_channels, gain=1.0, )
        self.style_shift_transform2 = EqualizedLinear(latent_size, out_channels, gain=1.0)

    def forward(self, x, latent1, latent2, noise=True):
        y = self.conv1(x)
        noise_shape = y.shape[0], 1, y.shape[2], y.shape[3]
        noise1 = torch.randn(noise_shape, device=x.device)
        y = y + self.noise_weight1 * noise1 if noise else y
        y = self.activation1(y)
        y = self.instance_norm1(y)
        y = self.style_scale_transform1(latent1)[:, :, None, None] * y \
            + self.style_shift_transform1(latent1)[:, :, None, None]
        y = self.conv2(y)
        noise2 = torch.randn(noise_shape, device=x.device)
        y = y + self.noise_weight2 * noise2 if noise else y
        y = self.activation2(y)
        y = self.instance_norm2(y)
        y = self.style_scale_transform2(latent2)[:, :, None, None] * y \
            + self.style_shift_transform2(latent2)[:, :, None, None]

        return y


class InputBlock(nn.Module):
    """
    One Block in Synthesis Net.

    Overall structure:
        Const. --> Add Noise --> Activation --> AdaIN --> Conv --> Add Noise --> Activation --> AdaIN

    """

    def __init__(self,
                 out_channels,
                 latent_size=w_dim,
                 ):
        super(InputBlock, self).__init__()
        self.const = nn.Parameter(torch.ones(1, out_channels, 4, 4))
        self.bias = nn.Parameter(torch.ones(out_channels))

        self.noise_weight1 = nn.Parameter(
            torch.randn((1, out_channels, 1, 1))
        )
        self.activation1 = nn.LeakyReLU(0.2)
        self.instance_norm1 = nn.InstanceNorm2d(out_channels)
        self.style_scale_transform1 = EqualizedLinear(latent_size, out_channels, gain=1.0)
        self.style_shift_transform1 = EqualizedLinear(latent_size, out_channels, gain=1.0)

        self.conv = EqualizedConv2d(out_channels, out_channels, kernel_size=3)
        self.noise_weight2 = nn.Parameter(
            torch.randn((1, out_channels, 1, 1))
        )
        self.activation2 = nn.LeakyReLU(0.2)
        self.instance_norm2 = nn.InstanceNorm2d(out_channels)
        self.style_scale_transform2 = EqualizedLinear(latent_size, out_channels, gain=1.0)
        self.style_shift_transform2 = EqualizedLinear(latent_size, out_channels, gain=1.0)

    def forward(self, latent1, latent2, noise=True):
        batch_size = latent1.size(0)
        x = self.const.expand(batch_size, -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        noise_shape = x.shape[0], 1, x.shape[2], x.shape[3]
        noise1 = torch.randn(noise_shape, device=x.device)
        y = x + self.noise_weight1 * noise1 if noise else x
        y = self.activation1(y)
        y = self.instance_norm1(y)
        y = self.style_scale_transform1(latent1)[:, :, None, None] * y \
            + self.style_shift_transform1(latent1)[:, :, None, None]
        y = self.conv(y)
        noise2 = torch.randn(noise_shape, device=x.device)
        y = y + self.noise_weight2 * noise2 if noise else y
        y = self.activation2(y)
        y = self.instance_norm2(y)
        y = self.style_scale_transform2(latent2)[:, :, None, None] * y \
            + self.style_shift_transform2(latent2)[:, :, None, None]

        return y


class SynthesisNet(nn.Module):
    """
    Synthesis Network.

    Args:
        latent_size: i.e. dimension of w
        num_channels: num of channels of output image, default: 3
        resolution: resoultion of output pic, default 1024
        fmap_base: Overall multiplier for the number of feature maps.
        fmap_decay: log2 feature map reduction when doubling the resolution.
        fmap_max: Maximum number of feature maps in any layer.

    """

    def __init__(self,
                 latent_size=w_dim,
                 num_channels=3,
                 resolution=1024,
                 fmap_base=8192,
                 fmap_decay=1.0,
                 fmap_max=512,
                 ):
        super().__init__()

        def nf(stage):
            """
            Given the stage, return the channel amount at this stage.
            """
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.blocks = {}
        self.to_rgb = {}
        stages = int(np.log2(resolution))
        self.blocks["block1"] = InputBlock(nf(1))
        self.to_rgb["after_block1"] = EqualizedConv2d(nf(1), num_channels, kernel_size=1)
        for stage in range(2, stages):
            self.blocks["block" + str(stage)] = SynthesisBlock(nf(stage - 1), nf(stage))
            self.to_rgb["after_block" + str(stage)] = EqualizedConv2d(nf(stage), num_channels, kernel_size=1)
        self.blocks = nn.ModuleDict(self.blocks)
        self.to_rgb = nn.ModuleDict(self.to_rgb)

    def forward(self,
                latents,
                stage,
                alpha,  # from 0 to 1
                noise=True
                ):
        output1 = self.blocks["block1"](latents[:, 0, :], latents[:, 1, :], noise)
        if stage == 1:
            return self.to_rgb["after_block1"](output1)
        for i in range(1, stage - 1):
            output1 = self.blocks["block" + str(i + 1)](output1, latents[:, 2 * i - 2, :], latents[:, 2 * i - 1, :],
                                                        noise)
        rgb1 = self.to_rgb["after_block" + str(stage - 1)](output1)
        rgb1 = nn.UpsamplingNearest2d(scale_factor=2)(rgb1)
        output2 = self.blocks["block" + str(stage)](output1, latents[:, 2 * stage - 2, :], latents[:, 2 * stage - 1, :],
                                                    noise)
        rgb2 = self.to_rgb["after_block" + str(stage)](output2)

        interpolation = alpha * rgb2 + (1 - alpha) * rgb1

        return interpolation


#######################
## Generator Network ##
#######################

class Generator(nn.Module):
    """
    Put Mapping Network and Synthesis Network together.

    """

    def __init__(self,
                 z_dim=512,
                 w_dim=512,
                 h_dim=512,
                 mapping_layers=8,
                 mapping_activation="LeakyReLU",
                 num_channels=3,
                 resolution=1024,
                 fmap_base=8192,
                 fmap_decay=1.0,
                 fmap_max=512,
                 truncation=True,
                 truncation_layer_cutoff=9,
                 ):
        super().__init__()
        self.mappingnet = MappingNet(z_dim, w_dim, h_dim, mapping_layers, mapping_activation,
                                     w_broadcast=2 * int(np.log2(resolution)) - 2)
        if truncation:
            self.truncation = Truncation(avg_latent=torch.zeros(w_dim),
                                         max_layer=truncation_layer_cutoff,
                                         beta=0.995)
        else:
            self.truncation = None
        self.synthesisnet = SynthesisNet(w_dim, 3, 1024, 8192, 1., 512)

    def forward(self, z, stage=9, alpha=1, truncation_psi=0.7, style_mix_prob=0.8, noise=True):

        w = self.mappingnet(z)

        if self.training:

            if style_mix_prob != 0.:
                z2 = torch.randn(z.shape).to(device)
                w2 = self.mappingnet(z2)

                if np.random.rand(1).item() < style_mix_prob:
                    mix_layer = np.random.randint(1, 2 * stage)
                    layer_idx = torch.from_numpy(np.arange(w.shape[1])[np.newaxis, :, np.newaxis]).to(device)
                    w_new = torch.where(layer_idx < mix_layer, w, w2)
                    w = w_new

            if self.truncation is not None:
                self.truncation.update(w[0, 0].detach())

        if (not self.training) and (self.truncation is not None):
            print("Truncating...")
            w = self.truncation(w, truncation_psi)

        output = self.synthesisnet.forward(w, stage, alpha, noise)

        return output



class Generator_without_noise_mapping(nn.Module):
    """
    Put Mapping Network and Synthesis Network together.

    """

    def __init__(self,
                 w_dim=512,
                 truncation=True,
                 truncation_layer_cutoff=9,
                 ):
        super().__init__()
        if truncation:
            self.truncation = Truncation(avg_latent=torch.zeros(w_dim),
                                         max_layer=truncation_layer_cutoff,
                                         beta=0.995)
        else:
            self.truncation = None
        self.synthesisnet = SynthesisNet(w_dim, 3, 1024, 8192, 1., 512)

    def forward(self, w, stage=9, alpha=1, truncation_psi=0.7, style_mix_prob=0.8, noise=True):


        if self.training:

            if style_mix_prob != 0.:
                z2 = torch.randn(z.shape).to(device)
                w2 = self.mappingnet(z2)

                if np.random.rand(1).item() < style_mix_prob:
                    mix_layer = np.random.randint(1, 2 * stage)
                    layer_idx = torch.from_numpy(np.arange(w.shape[1])[np.newaxis, :, np.newaxis]).to(device)
                    w_new = torch.where(layer_idx < mix_layer, w, w2)
                    w = w_new

            if self.truncation is not None:
                self.truncation.update(w[0, 0].detach())

        if (not self.training) and (self.truncation is not None):
            print("Truncating...")
            w = self.truncation(w, truncation_psi)

        output = self.synthesisnet.forward(w, stage, alpha, noise)

        return output


###########################
## Discriminator Network ##
###########################


class DiscriminatorBlock(nn.Module):
    """
    One Block in Disc Net.

    Overall structure:
        Conv --> Activation --> Blur --> Conv&Downsample --> Pooling --> Activation

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 blur_kernel=None,
                 ):
        super(DiscriminatorBlock, self).__init__()
        self.discblock = nn.Sequential(OrderedDict([
            ("conv1", EqualizedConv2d(in_channels, in_channels, kernel_size=3)),
            ("leakyrelu1", nn.LeakyReLU(0.2)),
            ("blur", BlurLayer(kernel=blur_kernel)),
            ("conv2", EqualizedConv2d(in_channels, out_channels, kernel_size=3, downscale=True)),
            ("leakyrelu2", nn.LeakyReLU(0.2)),
        ]))

    def forward(self, x):
        y = self.discblock(x)

        return y


class Discriminator(nn.Module):
    """
    Discriminator.

    Args:
        latent_size: i.e. dimension of w
        num_channels: num of channels of output image, default: 3
        resolution: resoultion of output pic, default 1024
        fmap_base: Overall multiplier for the number of feature maps.
        fmap_decay: log2 feature map reduction when doubling the resolution.
        fmap_max: Maximum number of feature maps in any layer.

    """

    def __init__(self,
                 num_channels=3,
                 resolution=1024,
                 fmap_base=8192,
                 fmap_decay=1.0,
                 fmap_max=512,
                 ):
        super(Discriminator, self).__init__()

        def nf(stage):
            """
            Given the stage, return the channel amount at this stage.
            """
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.blocks = {}
        self.from_rgb = {}
        stages = int(np.log2(resolution))
        for stage in range(stages - 1, 1, -1):
            self.from_rgb["before_block" + str(stage)] = EqualizedConv2d(num_channels, nf(stage), kernel_size=1)
            self.blocks["block" + str(stage)] = DiscriminatorBlock(nf(stage), nf(stage - 1))

        self.from_rgb["before_block1"] = EqualizedConv2d(num_channels, nf(1), kernel_size=1)
        self.msd = StddevLayer()
        self.blocks["block1"] = EqualizedConv2d(nf(1) + 1, nf(0), kernel_size=3)
        self.blocks = nn.ModuleDict(self.blocks)
        self.from_rgb = nn.ModuleDict(self.from_rgb)
        self.final = nn.Sequential(OrderedDict([
            ("leakyrelu1", nn.LeakyReLU(0.2)),
            ("linear1", EqualizedLinear(nf(1) * 4 * 4, nf(1))),
            ("leakyrelu2", nn.LeakyReLU(0.2)),
            ("linear2", EqualizedLinear(nf(1), 1, gain=1)),
        ]))

    def forward(self,
                image,
                stage=1,
                alpha=1,  # from 0 to 1
                ):
        if stage == 1:
            x = self.from_rgb["before_block1"](image)

        else:
            x1 = self.from_rgb["before_block" + str(stage)](image)
            x1 = self.blocks["block" + str(stage)](x1)
            x2 = self.from_rgb["before_block" + str(stage - 1)](nn.AvgPool2d(2)(image))

            x = alpha * x1 + (1 - alpha) * x2
            for i in range(stage - 1, 1, -1):
                x = self.blocks["block" + str(i)](x)

        x = self.msd(x)
        x = self.blocks["block1"](x)
        y = self.final(x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]))

        return y
