import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x
class SwinIR(nn.Module):
    def __init__(self, upscale, in_chans, img_size, window_size, img_range,
                 depths, embed_dim, num_heads, mlp_ratio, upsampler, resi_connection):
        super(SwinIR, self).__init__()

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.residual_layer = nn.Sequential(
            *[ResidualBlockNoBN(embed_dim) for _ in range(6)]
        )
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if upsampler == 'pixelshuffle':
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
            )
        else:
            self.upsample = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
    def forward(self, x):
        x = self.conv_first(x)
        res = self.residual_layer(x)
        res = self.conv_after_body(res)
        x = x + res
        x = self.upsample(x)
        return x
