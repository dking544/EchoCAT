import torch.nn as nn
import torch
import torch.nn.functional as F

def ReLU(inplace=True):
    return nn.ReLU(inplace=inplace)

def ReLU6(inplace=True):
    return nn.ReLU6(inplace=inplace)

def Sigmoid():
    return nn.Sigmoid()

def LeakyReLU(inplace=True):
    return nn.LeakyReLU(inplace=inplace)

def Tanh():
    return nn.Tanh()

class HSigmoid(nn.Module):
    def __init__(self, bias=3.0, divisor=6.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)

def HSwish(inplace=True):
    return nn.Hardswish(inplace=inplace)

class Swish(nn.Module):


    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class GELU(nn.Module):


    def forward(self, input):
        return F.gelu(input)
