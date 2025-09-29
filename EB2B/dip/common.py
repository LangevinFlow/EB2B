import torch
import torch.nn as nn
import numpy as np

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super().__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = [module(input) for module in self._modules.values()]
        shapes_h = [x.shape[2] for x in inputs]
        shapes_w = [x.shape[3] for x in inputs]

        if np.all(np.array(shapes_h) == min(shapes_h)) and np.all(np.array(shapes_w) == min(shapes_w)):
            inputs_ = inputs
        else:
            target_h = min(shapes_h)
            target_w = min(shapes_w)
            inputs_ = []
            for inp in inputs:
                diff_h = (inp.size(2) - target_h) // 2
                diff_w = (inp.size(3) - target_w) // 2
                inputs_.append(inp[:, :, diff_h: diff_h + target_h, diff_w: diff_w + target_w])
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super().__init__()
        self.dim2 = dim2

    def forward(self, input):
        size = list(input.size())
        size[1] = self.dim2
        noise = torch.zeros(size).type_as(input.data)
        noise.normal_()
        return torch.autograd.Variable(noise)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun='LeakyReLU'):
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        if act_fun == 'Swish':
            return Swish()
        if act_fun == 'ELU':
            return nn.ELU()
        if act_fun == 'none':
            return nn.Sequential()
        raise ValueError(f'Unknown activation {act_fun}')
    return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            raise ValueError('Unsupported downsample_mode')
        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = [x for x in (padder, convolver, downsampler) if x is not None]
    return nn.Sequential(*layers)
