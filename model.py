# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
import configs as configs
from torch.distributions.normal import Normal


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class ChannelAttention(nn.Module):
    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.input_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
        )
        self.out_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, data):
        inpt = self.input_conv(data)
        pool = self.global_avg_pool(inpt)
        weight = self.out_conv(pool)

        out = torch.multiply(inpt, weight)
        return out

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        self.config = config
        down_factor = config.down_factor
        patch_size = _triple(config.patches["size"])
        n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        self.patch_embeddings = Conv3d(in_channels=config.hidden_channels*4,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=1)


    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# transformer encoder block
class Encoder(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        down_factor = config.down_factor
        patch_size = _triple(config.patches["size"])
        self.hidden_size = config.hidden_size
        self.n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]) * (img_size[2]/2**down_factor// patch_size[2]))
        self.encoder_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.depth_sep_conv = DepthwiseSeparableConv(self.n_patches, self.n_patches)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for idx, layer_block in enumerate(self.layer):
            depth_conv = None
            if idx % 1 == 0:
                depth_conv = self.depth_sep_conv(hidden_states.reshape(1, self.n_patches, self.hidden_size, 1))
                depth_conv = torch.squeeze(depth_conv, -1)
            hidden_states, weights = layer_block(hidden_states)
            if depth_conv is not None:
                hidden_states = hidden_states + depth_conv*0.2
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.channel_attention = ChannelAttention(config.hidden_channels*4)
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, img_size, vis)

    def forward(self, input_ids):
        ca = self.channel_attention(input_ids)
        ca_output = self.embeddings(ca)
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        outputs = ca_output + encoded
        return outputs, attn_weights


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Trans2CNN(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.down_factor = config.down_factor
        self.img_size = img_size
        self.patch_size = _triple(config.patches["size"])

    def forward(self, transformer_feature):
        B, n_patch, hidden = transformer_feature.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        l, h, w = (self.img_size[0]//2**self.down_factor//self.patch_size[0]), (self.img_size[1]//2**self.down_factor//self.patch_size[1]), (self.img_size[2]//2**self.down_factor//self.patch_size[2])
        cnn_feature = transformer_feature.permute(0, 2, 1)
        cnn_feature = cnn_feature.contiguous().view(B, hidden, l, h, w)
        return cnn_feature

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class HCVTNet(nn.Module):
    def __init__(self, config, img_size=(64, 256, 256), int_steps=7, vis=False):
        super(HCVTNet, self).__init__()
        self.config = config
        self.transformer = Transformer(config, img_size, vis)
        self.trans2cnn = Trans2CNN(config, img_size)
        hidden_channels = config.hidden_channels
        self.reg_head = RegistrationHead(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=3,
        )

        # encoder
        self.inc = DoubleConv(1, hidden_channels)
        self.down1 = Down(hidden_channels, hidden_channels*2)
        self.down2 = Down(hidden_channels*2, hidden_channels*4)
        self.down3 = Down(hidden_channels*4, hidden_channels*4)
        self.down4 = Down(hidden_channels*4, hidden_channels*4)

        self.middle1 = nn.Conv3d(config.hidden_size, hidden_channels*4, kernel_size=3, stride=1, padding=1)
        self.middle2 = nn.Conv3d(hidden_channels*4, hidden_channels*4, kernel_size=3, stride=1, padding=1)

        # decoder
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.up4_conv = DecoderBlock(hidden_channels*4 + hidden_channels*4, hidden_channels*4)
        self.up3_conv = DecoderBlock(hidden_channels*4 + hidden_channels*4, hidden_channels*4)
        self.up2_conv = DecoderBlock(hidden_channels*4 + hidden_channels*4, hidden_channels*4)
        self.up1_conv = DecoderBlock(hidden_channels*4 + hidden_channels*2, hidden_channels*2)
        self.up_conv = DecoderBlock(hidden_channels*2 + hidden_channels, hidden_channels)

    def forward(self, x):
        input = self.inc(x)
        down1 = self.down1(input)
        down2 = self.down2(down1)

        trans_feature, attn_weights = self.transformer(down2)
        cnn_feature = self.trans2cnn(trans_feature)

        down3 = self.down3(down2)
        down4 = self.down4(down3)

        middle = self.middle1(cnn_feature)
        middle = self.middle2(middle)

        up4 = self.up(middle)
        merge4 = torch.cat([up4, down4], dim=1)
        up4_conv = self.up4_conv(merge4)

        up3 = self.up(up4_conv)
        merge3 = torch.cat([up3, down3], dim=1)
        up3_conv = self.up3_conv(merge3)

        up2 = self.up(up3_conv)
        merge2 = torch.cat([up2, down2], dim=1)
        up2_conv = self.up2_conv(merge2)

        up1 = self.up(up2_conv)
        merge1 = torch.cat([up1, down1], dim=1)
        up1_conv = self.up1_conv(merge1)

        up = self.up(up1_conv)
        merge = torch.cat([up, input], dim=1)
        up_conv = self.up_conv(merge)

        out = self.reg_head(up_conv)
        return out



CONFIGS = {
    'HCVT-Net': configs.get_config(),
}

