import torch
import torch.nn as nn
from model.GLU import MyGLU
from model.CBAM import CBAM
from model.MHSA import MHSA
from model.CGRUC import CGRUC

class ParallelBlock(nn.Module):
    def __init__(self, in_ch,in_L):
        super(ParallelBlock, self).__init__()
        self.glu = MyGLU(in_ch=in_ch)
        self.cbam = CBAM(in_ch= in_ch)
        self.out = nn.Sequential(
            nn.LayerNorm(in_L),
            nn.Conv2d(in_channels=in_ch*2,out_channels=in_ch,kernel_size=1,stride=1,padding=0),
        )

    def forward(self, x):
        y1 = self.glu(x)
        y2 = self.cbam(x)
        out = torch.cat((y1,y2),dim=1)
        out = self.out(out)
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch,in_L):
        super(BottleneckBlock, self).__init__()
        self.layer_1 = CGRUC(in_L=in_L, in_ch=in_ch, factor=4)
        self.layer_2 = MHSA(d_model=in_ch*in_L, num_heads=8, dropout_p=0.05)
        self.out = nn.LayerNorm(in_L)

    def forward(self, x):
        y1 = self.layer_1(x)
        y2 = self.layer_2(x)
        out = torch.cat((y1,y2),dim=1)
        out = self.out(out)
        return out

class Encoder(nn.Module):
    def __init__(self, kernel_size, layers, in_len):
        super().__init__()
        self.kernel_size = kernel_size
        self.layers = layers
        self.in_len = in_len
        self.in_channels = 1
        self.out_channels = [64,64,64,64,64,64]
        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels[0],
                      kernel_size=1, stride=1,padding=0),
            ParallelBlock(in_ch=self.out_channels[0],in_L=self.in_len[0]*2),
        )
        self.encoder_layers = nn.ModuleList()
        for idx in range(self.layers):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.out_channels[idx], out_channels=self.out_channels[idx+1],
                              kernel_size=(1,self.kernel_size[idx]),
                              stride=(1,2),
                              padding=(0,(self.kernel_size[idx] - 1) // 2)),
                    nn.PReLU(self.out_channels[idx+1]),
                    nn.LayerNorm(self.in_len[idx]),
                    ParallelBlock(in_ch=self.out_channels[idx+1],in_L=self.in_len[idx]),
                )
            )
        self.bottleneck = BottleneckBlock(in_ch=self.out_channels[idx+1], in_L= self.in_len[idx])

    def forward(self, x):
        x = self.in_block(x)
        skip = []
        for idx in range(self.layers-1):
            x = self.encoder_layers[idx](x)
            skip.append(x)
        x = self.encoder_layers[-1](x)
        x = self.bottleneck(x)
        return x, skip

class Decoder(nn.Module):
    def __init__(self, kernel_size, layers, in_len):
        super().__init__()
        self.in_len = (torch.tensor(in_len[::-1])*2).tolist()
        self.layers = layers
        self.kernel_size = kernel_size
        self.out_channels = [64,64,64,64,64,64]
        self.out_channels = self.out_channels[::-1]
        self.decoder_layers = nn.ModuleList()
        for idx in range(self.layers):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.out_channels[idx]*2,
                                       out_channels=self.out_channels[idx+1],
                                       kernel_size=(1,self.kernel_size[idx]),
                                       stride=(1,2),
                                       padding=(0,(self.kernel_size[idx] - 1) // 2),
                                       output_padding=(0,1)),
                    nn.PReLU(self.out_channels[idx+1]),
                    nn.LayerNorm(self.in_len[idx]),
                    ParallelBlock(in_ch=self.out_channels[idx+1],in_L=self.in_len[idx]),
                )
            )

        self.out_block = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels[idx+1], out_channels=1,
                      kernel_size=1, stride=1,padding=0),
            nn.Tanh()
        )

    def forward(self, x, skip):
        skip = skip[::-1]
        x = self.decoder_layers[0](x)
        for idx in range(1,self.layers):
            x = torch.cat([x, skip[idx-1]], dim=1)
            x = self.decoder_layers[idx](x)
        x = self.out_block(x)
        return x

class PUNET(nn.Module):
    def __init__(self,):
        super().__init__()
        self.kernel_size = [11,11,11,11,11]
        self.in_len = [256,128,64,32,16]
        self.layers = len(self.in_len)
        self.encoder = Encoder(kernel_size=self.kernel_size, layers = self.layers,in_len=self.in_len)
        self.decoder = Decoder(kernel_size=self.kernel_size, layers = self.layers,in_len=self.in_len)

    def forward(self, x):
        x, skip = self.encoder(x)
        x = self.decoder(x, skip)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    batch_size = 8
    in_channel = 1
    fs = int(16e3)
    T = 256
    L = 512
    model = PUNET()
    summary(model, input_size=[batch_size, in_channel, T, L])
