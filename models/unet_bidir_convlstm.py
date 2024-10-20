import torch
import torch.nn as nn
from collections import OrderedDict
from models.blocks import ConvLSTMCell2D


class UNet2DConvLSTM(nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32
    ):
        super(UNet2DConvLSTM, self).__init__()
        self.init_features = init_features

        # Encoder layers
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

        # Bottleneck
        self.bottleneck = UNet2DConvLSTM._block(
            init_features * 8, init_features * 16, name="bottleneck"
        )

        # ConvLSTM
        self.conv_lstm = ConvLSTMCell2D(
            16 * init_features, 16 * init_features, kernel_size=3
        )

        # 1x1 Conv to reduce channels after concatenation of LSTM states
        self.reduce_channels = nn.Conv2d(
            16 * init_features * 2, 16 * init_features, kernel_size=1
        )

        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(
            init_features * 16, init_features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2DConvLSTM._block(
            init_features * 16, init_features * 8, name="dec4"
        )

        self.upconv3 = nn.ConvTranspose2d(
            init_features * 8, init_features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2DConvLSTM._block(
            init_features * 8, init_features * 4, name="dec3"
        )

        self.upconv2 = nn.ConvTranspose2d(
            init_features * 4, init_features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2DConvLSTM._block(
            init_features * 4, init_features * 2, name="dec2"
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

        # Initialize LSTM hidden states
        h_fwd, c_fwd = self.conv_lstm.init_hidden(
            (x.shape[0], 16 * self.init_features, 16, 16)
        )
        h_bwd, c_bwd = self.conv_lstm.init_hidden(
            (x.shape[0], 16 * self.init_features, 16, 16)
        )

        # Forward and Backward pass combined
        for t in range(num_slices // 2 + 1):
            slice_fwd = x[:, t].unsqueeze(1)  # Forward slice
            slice_bwd = x[:, num_slices - 1 - t].unsqueeze(1)  # Backward slice

            # Encoding forward
            enc1_fwd = self.encoder1(slice_fwd)
            enc2_fwd = self.encoder2(self.pool1(enc1_fwd))
            enc3_fwd = self.encoder3(self.pool2(enc2_fwd))
            enc4_fwd = self.encoder4(self.pool3(enc3_fwd))
            bottleneck_fwd = self.bottleneck(self.pool4(enc4_fwd))

            # Encoding backward
            enc1_bwd = self.encoder1(slice_bwd)
            enc2_bwd = self.encoder2(self.pool1(enc1_bwd))
            enc3_bwd = self.encoder3(self.pool2(enc2_bwd))
            enc4_bwd = self.encoder4(self.pool3(enc3_bwd))
            bottleneck_bwd = self.bottleneck(self.pool4(enc4_bwd))

            # ConvLSTM forward and backward
            h_fwd, c_fwd = self.conv_lstm(bottleneck_fwd, (h_fwd, c_fwd))
            h_bwd, c_bwd = self.conv_lstm(bottleneck_bwd, (h_bwd, c_bwd))

        # Concatenate forward and backward LSTM outputs
        h_combined = torch.cat([h_fwd, h_bwd], dim=1)

        # Reduce channels back to the desired number
        h_reduced = self.reduce_channels(h_combined)

        # Decoder
        dec4 = self.upconv4(h_reduced)
        dec4 = torch.cat((dec4, enc4_fwd), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3_fwd), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2_fwd), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1_fwd), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

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
