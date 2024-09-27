import torch
import torch.nn as nn
from collections import OrderedDict
from torch import nn
import torch
from models.blocks import ConvLSTMCell2D


class UNet2DConvLSTM(nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32
    ):
        super(UNet2DConvLSTM, self).__init__()
        self.init_features = init_features
        self.encoder1 = UNet2DConvLSTM._block(in_channels, init_features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2DConvLSTM._block(
            init_features, init_features * 2, name="enc2"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2DConvLSTM._block(
            init_features * 2, init_features * 4, name="enc3"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2DConvLSTM._block(
            init_features * 4, init_features * 8, name="enc4"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2DConvLSTM._block(
            init_features * 8, init_features * 16, name="bottleneck"
        )

        self.conv_lstm = ConvLSTMCell2D(16 * init_features, 16 * init_features, 3)

        self.upconv4 = nn.ConvTranspose2d(
            init_features * 16, init_features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2DConvLSTM._block(
            (init_features * 8) * 2, init_features * 8, name="dec4"
        )

        self.upconv3 = nn.ConvTranspose2d(
            init_features * 8, init_features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2DConvLSTM._block(
            (init_features * 4) * 2, init_features * 4, name="dec3"
        )

        self.upconv2 = nn.ConvTranspose2d(
            init_features * 4, init_features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2DConvLSTM._block(
            (init_features * 2) * 2, init_features * 2, name="dec2"
        )

        self.upconv1 = nn.ConvTranspose2d(
            init_features * 2, init_features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2DConvLSTM._block(
            init_features * 2, init_features, name="dec1"
        )

        self.conv = nn.Conv2d(
            in_channels=init_features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_slices = x.shape[1]
        enc1_seq, enc2_seq, enc3_seq, enc4_seq, bn_seq = [], [], [], [], []
        for t in range(num_slices):
            enc1 = self.encoder1(x[:, 1].unsqueeze(1))
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))
            bottleneck = self.bottleneck(self.pool4(enc4))

            enc1_seq.append(enc1)
            enc2_seq.append(enc2)
            enc3_seq.append(enc3)
            enc4_seq.append(enc4)
            bn_seq.append(bottleneck)

        h, c = self.conv_lstm.init_hidden(bn_seq[0].shape)

        lstm_out = []
        for t in range(num_slices):
            h, c = self.conv_lstm(bn_seq[t], (h, c))
            lstm_out.append(h)

        outputs = []
        for t in range(num_slices):
            dec4 = self.upconv4(lstm_out[t])
            dec4 = torch.cat((dec4, enc4_seq[t]), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3_seq[t]), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2_seq[t]), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1_seq[t]), dim=1)
            dec1 = self.decoder1(dec1)
            outputs.append(torch.sigmoid(self.conv(dec1)))

        return torch.cat(outputs, dim=1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "dropout1", nn.Dropout2d(p=0.15)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
