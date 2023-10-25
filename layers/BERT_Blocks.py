import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class ConformerEncoderLayer(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        depthwise_conv_kernel_size=31,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
        """
        super(ConformerEncoderLayer, self).__init__()

        self.ffn1 = PositionwiseFeedForward(
            embed_dim,
            ffn_embed_dim,
            dropout,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.self_attn = MultiHeadedAttention(
            embed_dim,
            attention_heads,
            dropout=dropout,
        )

        self.conv_module = ConvolutionModule(
            embed_dim=embed_dim,
            channels=embed_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

        self.ffn2 = PositionwiseFeedForward(
            embed_dim,
            ffn_embed_dim,
            dropout,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
    ):
        """
        Args:
            x: Tensor of shape B x nvar x T x d_model
            encoder_padding_mask: Optional mask tensor
        Returns:
            Tensor of shape B x nvar x T x d_model
        """
        residual = x 
        x = self.ffn1(x) 
        x = x * 0.5 + residual 
        residual = x 
        x = self.self_attn_layer_norm(x) 
        x = self.self_attn(
            query=x.view(-1, *(x.shape[-2:])),
            key=x.view(-1, *(x.shape[-2:])),
            value=x.view(-1, *(x.shape[-2:])),
            key_padding_mask=encoder_padding_mask,
        ).view_as(residual)
        x = self.self_attn_dropout(x)
        x = x + residual 

        residual = x
        x = self.conv_module(x.view(-1, *(x.shape[-2:]))).view_as(residual)
        x = residual + x

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x
    

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    def forward(self, x):
        return self.dropout(self.w_2(self.dropout(self.activation(self.w_1(x)))))


class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
        self,
        embed_dim,
        channels,
        depthwise_kernel_size,
        dropout,
        bias=False,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            bias: If bias should be added to conv layers
        """
        super(ConvolutionModule, self).__init__()
        assert (
            depthwise_kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.activation = nn.SiLU()
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        x = self.layer_norm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)
    

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        dropout: Dropout rate.
    """

    def __init__(self, n_feat, n_head, dropout):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward_qkv(self, query, key, value, **kwargs):
        """Transform query, key and value.
        Args:
            query: Query tensor  B X T1 X C
            key: Key tensor B X T2 X C
            value: Value tensor  B X T2 X C
        Returns:
            torch.Tensor: Transformed query tensor  B X n_head X T1 X d_k
            torch.Tensor: Transformed key tensor B X n_head X T2 X d_k
            torch.Tensor: Transformed value tensor  B X n_head X T2 X d_k
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value: Transformed value B X n_head X T2 X d_k.
            scores: Attention score  B X n_head X T1 X T2
            mask: Mask  T2 X B
        Returns:
            torch.Tensor: Transformed value  B X T1 X d_model
                weighted by the attention score  B X T1 X T2
        """
        n_batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2).to(bool),
                float("-inf"),  # (batch, head, time1, time2)
            )
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor T X B X C
            key (torch.Tensor): Key tensor T X B X C
            value (torch.Tensor): Value tensor T X B X C
            mask (torch.Tensor): Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores