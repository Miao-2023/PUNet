import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_ch=64, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))
        max_out =self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=11):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1,
                               kernel_size=(1,kernel_size), stride=1,
                               padding=(0, (kernel_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_ch):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

if __name__ == '__main__':
    from torchinfo import summary
    batch_size = 2
    in_channel = 64
    T = 256
    L = 512
    model = CBAM(in_ch=in_channel)
    summary(model, input_size=[batch_size, in_channel, T, L])