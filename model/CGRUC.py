import torch
import torch.nn as nn

class CGRUC(nn.Module):
    def __init__(self, in_L, in_ch, factor=16):
        super(CGRUC, self).__init__()
        self.in_L = in_L
        self.in_ch = in_ch
        self.dropout = 0.05
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=in_ch//factor,kernel_size=1,stride=1,padding=0),
        )
        self.gru = nn.GRU(input_size=self.in_L*self.in_ch // factor,hidden_size=self.in_L*self.in_ch // factor,
                          num_layers=2, batch_first=True,dropout=0.05,bidirectional=True)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch//factor,out_channels=in_ch,kernel_size=1,stride=1,padding=0),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = x.transpose(1,2).contiguous()
        shape = x.shape
        x = x.view((shape[0],shape[1],-1))
        (x,_) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]
        x = x.view(shape)
        x = x.transpose(1,2).contiguous()
        x = self.conv_2(x)
        return x

if __name__ == '__main__':
    from torchinfo import summary
    batch_size = 16*16
    in_channel = 64
    T = 256
    L = 16
    model = CGRUC(in_L = L, in_ch = in_channel, factor=4)
    summary(model, input_size=[batch_size, in_channel, T, L])