import torch.nn as nn

from .common import Concat, act, bn, conv
from .non_local_dot_product import NONLocalBlock2D


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduction = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore[override]
        weight = self.fc(self.pool(x))
        return x * weight


class ResidualCAB(nn.Module):
    def __init__(self, channels: int, act_fun: str = 'LeakyReLU', pad: str = 'zero', reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv(channels, channels, 3, bias=True, pad=pad),
            bn(channels),
            act(act_fun),
            conv(channels, channels, 3, bias=True, pad=pad),
            bn(channels),
        )
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.activation = act(act_fun)

    def forward(self, x):  # type: ignore[override]
        res = self.body(x)
        res = self.ca(res)
        res = res + x
        return self.activation(res)


def skip(num_input_channels=2, num_output_channels=3,
         num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
         num_channels_skip=[4, 4, 4, 4, 4],
         filter_size_down=3, filter_size_up=3, filter_skip_size=1,
         need_sigmoid=True, need_bias=True,
         pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
         need1x1_up=True):
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not isinstance(upsample_mode, (list, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not isinstance(downsample_mode, (list, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    if not isinstance(filter_size_down, (list, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not isinstance(filter_size_up, (list, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(n_scales):
        deeper = nn.Sequential()
        skip_model = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_model, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip_model.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip_model.add(bn(num_channels_skip[i]))
            skip_model.add(act(act_fun))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        if i > 1:
            deeper.add(NONLocalBlock2D(in_channels=num_channels_down[i]))

        deeper.add(ResidualCAB(num_channels_down[i], act_fun=act_fun, pad=pad))

        deeper_main = nn.Sequential()

        if i == n_scales - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias,
                           pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        model_tmp.add(ResidualCAB(num_channels_up[i], act_fun=act_fun, pad=pad))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
