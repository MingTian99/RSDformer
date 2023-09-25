import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from thop import profile

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class InvertedBottleneckBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio, padding, dilation):
        super(InvertedBottleneckBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            LayerNorm(hidden_dim, LayerNorm_type='WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=padding, groups=hidden_dim, bias=False),
            LayerNorm(hidden_dim, LayerNorm_type='WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            LayerNorm(oup, LayerNorm_type='WithBias'),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, dim):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedBottleneckBlock(inp=dim // 2, oup=dim // 2, dilation=1, padding=1, expand_ratio=2)
        self.theta_rho = InvertedBottleneckBlock(inp=dim // 2, oup=dim // 2, dilation=1, padding=1, expand_ratio=2)
        self.theta_eta = InvertedBottleneckBlock(inp=dim // 2, oup=dim // 2, dilation=1, padding=1, expand_ratio=2)
        self.shffleconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        l = self.shffleconv(torch.cat((z1, z2), dim=1))
        z1, z2 = self.separateFeature(l)
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtractionBlock(nn.Module):
    def __init__(self, dim, num_layers=3):
        super(DetailFeatureExtractionBlock, self).__init__()
        INNmodules = [DetailNode(dim) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x.chunk(2, dim=1)
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class LowPassFilter(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU(True)
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.interpolate(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear', align_corners=False) for i
                  in range(4)]
        bottle = torch.cat(priors, 1)

        return self.relu(bottle)


class HighPassFilter(nn.Module):
    def __init__(self, in_channel, sizes=(1, 4, 8, 16)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU(True)
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveMaxPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.interpolate(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear', align_corners=False) for i
                  in range(4)]
        bottle = torch.cat(priors, 1)

        return self.relu(bottle)


##########################################################################
# Dynamics_Filter_Module
class Dynamics_Filter_Module(nn.Module):
    def __init__(self, dim=32, bias=False):
        super(Dynamics_Filter_Module, self).__init__()
        self.LP = LowPassFilter(in_channel=dim)
        self.HP = HighPassFilter(in_channel=dim)
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_1 = self.LP(x)
        x_2 = self.HP(x)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.fusion(x)
        return x


##########################################################################
# Dynamic_Gated_Fusion_Block
class Dynamic_Gated_Fusion_Block(nn.Module):
    def __init__(self, m=-0.80):
        super(Dynamic_Gated_Fusion_Block, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

##########################################################################
# Dual_Frequency_Adaptive_Block
class Dual_Frequency_Adaptive_Block(nn.Module):
    def __init__(self, dim, mlp_ratio, bias, LayerNorm_type):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1)
        self.fc2 = nn.Conv2d(hidden_features, dim, 1)
        self.act = nn.ReLU(True)
        self.dynamices_filters = Dynamics_Filter_Module(dim=hidden_features, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.fc1(x)
        x = x + self.act(self.dynamices_filters(x))
        x = self.fc2(x)
        return x


##########################################################################
# Detail_Compensated_Transpose_Attention
class Detail_Compensated_Transpose_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Detail_Compensated_Transpose_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.DFEB = DetailFeatureExtractionBlock(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        y = self.DFEB(x)
        out = self.project_out(torch.cat([out, y], dim=1))
        return out


##########################################################################
# Remote_Sensing_Transformer_Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Detail_Compensated_Transpose_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = Dual_Frequency_Adaptive_Block(dim, ffn_expansion_factor, bias, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Remote_Sensing_Dehazing_Transformer
class Remote_Sensing_Dehazing_Transformer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=[24, 48, 96, 192], num_blocks=[2, 4, 4, 2],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias', ):
        super(Remote_Sensing_Dehazing_Transformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim[0])
        self.encoder_level1 = nn.Sequential(*[

            TransformerBlock(dim=dim[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, ) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim[0])  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim[1]), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim[1]))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim[2]), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim[2]))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim[2]), int(dim[1]), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim[1]), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim[1]))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.DGFB1 = Dynamic_Gated_Fusion_Block(m=-1)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim[0]), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim[0]), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[0])])
        self.DGFB2 = Dynamic_Gated_Fusion_Block(m=-0.6)
        self.output = nn.Conv2d(int(dim[0]), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_dec_level2 = self.up3_2(out_enc_level3)
        inp_dec_level2 = self.DGFB1(inp_dec_level2, out_enc_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.DGFB2(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


def RSDformer():
    return Remote_Sensing_Dehazing_Transformer(inp_channels=3, out_channels=3, dim=[24, 48, 96, 192],
                                               num_blocks=[4, 8, 8, 4], heads=[2, 4, 8, 8],
                                               ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias')

