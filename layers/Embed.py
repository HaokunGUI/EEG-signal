import torch
import torch.nn as nn
import math


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
    
class Tokenizer(nn.Module):
    def __init__(self, in_channel: int, patch_size: int, embedding_dim: int, kernel_num: int=5, hidden_dim:int=4):
        """
        Initializes a Tokenizer module for processing input data.

        Args:
            in_channel (int): The number of input channels in the time series data.
            patch_size (int): The size of the data patches to be processed.
            embedding_dim (int): The dimension of the output embeddings.
        """
        super(Tokenizer, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channel = in_channel
        self.kernel_num = kernel_num
        self.bottleneck_conv1d = nn.Conv1d(in_channel, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.max_pooling = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1d = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, in_channel, kernel_size=2*i+1, stride=1, padding=i, bias=False) for i in range(kernel_num)
            ]
        )        
        self.max_conv = nn.Conv1d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_1 = nn.Conv1d(in_channel*(kernel_num+1), embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.deepwise_conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=patch_size, stride=1, padding=0, groups=embedding_dim, bias=False)
        self.conv_2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        assert T % self.patch_size == 0, 'Input sequence length must be divisible by the patch size.'
        patch_num = T // self.patch_size
        x = x.reshape(B, patch_num, self.patch_size, C).permute(0, 1, 3, 2).reshape(-1, C, self.patch_size)

        # x [B*T, C, D]
        conv_list = []
        max_pooling = self.max_conv(self.max_pooling(x))
        conv_list.append(max_pooling)
        x_inception = self.bottleneck_conv1d(x)
        for i in range(self.kernel_num):
            conv_list.append(self.conv1d[i](x_inception))
        x = torch.cat(conv_list, dim=1)

        x = self.conv_1(x)
        x = self.deepwise_conv(x)
        x = self.conv_2(x)

        x = x.squeeze(-1).reshape(B, patch_num, -1)
        x = self.layer_norm(x)
        return x


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding module (new implementation).

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(self, max_len, d_model):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x : Input tensor B x T x C.
        Returns:
            torch.Tensor: Encoded tensor B x T x C.

        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb
