import torch.nn as nn
from torch import Tensor

class PointwiseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ):
        super(PointwiseConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class MyGLU(nn.Module):
    def __init__(self, in_ch, dim = 1):
        super(MyGLU, self).__init__()
        self.conv = PointwiseConv2d(in_channels = in_ch, out_channels = in_ch*2,
                                    stride = 1, padding = 0, bias = True)
        self.dim = dim
        self.GLU = GLU(self.dim)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.conv(inputs)
        outputs = self.GLU(outputs)
        return outputs

if __name__ == '__main__':
    from torchinfo import summary

    batch_size = 2
    in_channel = 64
    fs = int(16e3)
    T = 32
    L = 2048
    model = MyGLU(in_channel)
    summary(model, input_size=[batch_size, in_channel, T, L])  # 输出网络结构 [B, C, N]