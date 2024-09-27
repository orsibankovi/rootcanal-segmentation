import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super(EncoderBlock2D, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x, x


class DecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super(DecoderBlock2D, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SliceAttention(nn.Module):
    def __init__(self, embed_dim: int, num_slices: int):
        super(SliceAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_slices = num_slices

        self.query_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, num_slices, embed_dim)
        batch_size, num_slices, embed_dim = x.size()

        # Transpose to (batch_size, embed_dim, num_slices)
        x = x.transpose(1, 2)

        Q = self.query_conv(x)  # (batch_size, embed_dim, num_slices)
        K = self.key_conv(x)  # (batch_size, embed_dim, num_slices)
        V = self.value_conv(x)  # (batch_size, embed_dim, num_slices)

        # Compute attention scores
        scores = torch.bmm(Q.transpose(1, 2), K) / (
            embed_dim**0.5
        )  # (batch_size, num_slices, num_slices)
        attention_weights = self.softmax(scores)  # (batch_size, num_slices, num_slices)

        # Compute the weighted sum of values
        out = torch.bmm(
            V, attention_weights.transpose(1, 2)
        )  # (batch_size, embed_dim, num_slices)

        # Transpose back to (batch_size, num_slices, embed_dim)
        out = out.transpose(1, 2)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert (
            self.head_dim * num_heads == dim
        ), "Embedding dimension must be divisible by the number of heads"

        # Linear layers for Q, K, V projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape

        # Compute Q, K, V
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)  # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)

        # Reshape Q, K, V to (batch_size, num_heads, seq_length, head_dim)
        Q = Q.view(batch_size, self.num_heads, seq_length, self.head_dim)
        K = K.view(batch_size, self.num_heads, seq_length, self.head_dim)
        V = V.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # Scaled Dot-Product Attention
        attention_scores = torch.einsum("bhqd, bhkd -> bhqk", Q, K) / (
            self.head_dim**0.5
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_out = torch.einsum("bhqk, bhvd -> bhqd", attention_weights, V)

        # Reshape and concatenate the heads
        attention_out = attention_out.view(batch_size, seq_length, embed_dim)

        # Final linear layer
        out = self.fc_out(attention_out)

        return out


class ConvLSTMCell2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell2D, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Convolutional gates
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, shape):
        h = torch.zeros(*shape, device=self.conv.weight.device)
        c = torch.zeros(*shape, device=self.conv.weight.device)
        return h, c
