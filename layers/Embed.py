import torch
import torch.nn as nn
import math


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


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class Tokenizer(nn.Module):
    def __init__(self, in_channel: int, patch_size: int, embedding_dim: int, shared_embedding: bool = False):
        """
        Initializes a Tokenizer module for processing input data.

        Args:
            in_channel (int): The number of input channels in the time series data.
            patch_size (int): The size of the data patches to be processed.
            embedding_dim (int): The dimension of the output embeddings.
            shared_embedding (bool, optional): If True, share the embedding layer across input channels.
                Defaults to False.
        """
        super(Tokenizer, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channel = in_channel
        self.share_embedding = shared_embedding
        # Define a linear layer for encoding the input data.
        if not shared_embedding:
            self.encoding = nn.ModuleList()
            for _ in range(in_channel):
                self.encoding.append(nn.Linear(patch_size, embedding_dim))
        else:
            self.encoding = nn.Linear(patch_size, embedding_dim)

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the Tokenizer.

        Args:
            input (torch.Tensor): The input data tensor with dimensions (B, S, D), where
                B is the batch size, S is the sequence length, and D is the input data dimension.

        Returns:
            torch.Tensor: The encoded data tensor with dimensions (B, D, patch_num, embedding_dim), where
                B is the batch size, D is the input data dimension, patch_num is the num of the patch block 
                and embedding_dim is the specified embedding dimension.
        """
        B, S, D = input.shape
        assert S % self.patch_size == 0, 'Input sequence length must be divisible by the patch size.'

        # Calculate the number of patches.
        patch_num = S // self.patch_size

        # Reshape the input into patches, resulting in a tensor with dimensions (B, D, patch_num, patch_size).
        sequence = input.reshape(B, patch_num, self.patch_size, D).permute(0, 3, 1, 2)

        # Apply the linear layer (encoding) to the sequence.
        if not self.share_embedding:
            x_out = []
            for i in range(self.in_channel):
                x_out.append(self.encoding[i](sequence[:, i, :, :]))
            encoding = torch.stack(x_out, dim=1)
        else:
            encoding = self.encoding(sequence)

        return encoding
