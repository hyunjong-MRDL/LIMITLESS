import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class Down(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.block = ConvBlock(in_channel,
                               out_channel)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.block(x)

        return x
    
# Decoder
class Up(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.block = ConvBlock(in_channel,
                               out_channel,
                               mid_channels = in_channel // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2 ,x1], dim=1) 

        return self.block(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Model(nn.Module):
    def __init__(self,
                 args,
                 in_channel,
                 out_channel):
        super().__init__()
        self.args = args
        width_multiplier = 1
        _channels = (32, 64, 128, 256, 512)

        self.n_channels = _channels
        self.out_channel = out_channel

        self.channels = [int(c*width_multiplier) for c in _channels]
        
        factor = 2

        self.inc = ConvBlock(in_channel, self.channels[0])
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])
        self.down4= Down(self.channels[3], self.channels[4] // factor)

        self.up1 = Up(self.channels[4], self.channels[3] // factor)
        self.up2 = Up(self.channels[3], self.channels[2] // factor)
        self.up3 = Up(self.channels[2], self.channels[1] // factor)
        self.up4 = Up(self.channels[1], self.channels[0])
        self.outc = OutConv(self.channels[0], out_channel)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        print(x1.shape)
        x2 = self.down1(x1)
        print(x2.shape)
        x3 = self.down2(x2)
        print(x3.shape)
        x4 = self.down3(x3)
        print(x4.shape)
        x5 = self.down4(x4)
        print(x5.shape)

        # decoder
        x1_d = self.up1(x5, x4)
        print(x1_d.shape)
        x2_d = self.up2(x1_d, x3)
        print(x2_d.shape)
        x3_d = self.up3(x2_d, x2)
        print(x3_d.shape)
        x4_d = self.up4(x3_d, x1)
        print(x4_d.shape)

        pred = self.outc(x4_d)

        return pred
    
class PModel(nn.Module):
    def __init__(self,
                 args,
                 in_channel,
                 out_channel):
        super().__init__()

        model1 = Model(args,
                    in_channel=27,
                    out_channel=1)
        model2 = Model(args,
                    in_channel=27,
                    out_channel=1)
        model3 = Model(args,
                    in_channel=27,
                    out_channel=1)
        
    def forward(self, A, P, D):
        return A, P, D
    
# build_model
def build_model(args):
    model = Model(args,
                  in_channel=27,
                  out_channel=27)
    
    return model

if __name__=='__main__':
    model = build_model(args = 1)
    sample = torch.randn(4, 27, 32, 128, 128)

    print(f"Model Parameter : {sum(p.numel() for p in model.parameters())}")
    pred = model(sample)
    print(pred.shape)