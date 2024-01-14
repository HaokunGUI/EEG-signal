import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

# class Trans_Conv(nn.Module):
#     def __init__(self, d_model: int, dropout: float, in_channels: int, out_channels: int, activation: str='gelu'):
#         super(Trans_Conv, self).__init__()
#         self.transformer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=8,
#             dim_feedforward=4*d_model,
#             dropout=dropout,
#             activation=activation,
#             batch_first=True,
#         )
#         self.conv = nn.Conv1d(in_channels=in_channels, 
#                               out_channels=out_channels, 
#                               kernel_size=1
#                             )
#         self.activation = self._get_activation_fn(activation)
    
#     def forward(self, x: torch.Tensor):
#         # x: (B, C, T, d_model)
#         B, C, T, D = x.shape
#         x = x.contiguous().view(B*C, T, D) # (B*C, T, d_model)
#         x = self.transformer(x) # (B*C, T, d_model)
#         x = x.view(B, C, T, D) # (B, C, T, d_model)
#         y = x.transpose(1, 2)
#         y = y.contiguous().view(-1, C, D) # (B*T, C, d_model)
#         y = self.conv(y) # (B*T, C', d_model)
#         y = y.view(B, T, -1, D)
#         y = y.transpose(1, 2) # (B, C', T, d_model)

#         # y = x.transpose(1, 2) # (B, T, C, d_model)
#         # y = y.contiguous().view(-1, C, D)
#         # y = self.agg(y)
#         # y = y.view(B, T, -1, D) # (B, T, C, d_model)
#         # y = y.transpose(1, 2) # (B, C, T, d_model)

#         y = self.activation(y) # (B, C', T, d_model)
#         return y
    
#     @staticmethod
#     def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
#         if activation == "relu":
#             return F.relu
#         elif activation == "gelu":
#             return F.gelu

#         raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    
# class Trans_Conv(nn.Module):
#     def __init__(self, d_model: int, dropout: float, in_channels: int, out_channels: int, activation: str='gelu',
#                  **kwargs):
#         super(Trans_Conv, self).__init__(**kwargs)

#         self.attn_1 = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=8,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.attn_2 = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=8,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, 4*d_model),
#             self._get_activation_fn(activation),
#             nn.Dropout(dropout),
#             nn.Linear(4*d_model, d_model),
#         )

#         self.ln_1 = nn.LayerNorm(d_model)
#         self.ln_2 = nn.LayerNorm(d_model)
#         self.ln_3 = nn.LayerNorm(d_model)

#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)
#         self.dropout_3 = nn.Dropout(dropout)

#         self.activation = self._get_activation_fn(activation)

    
#     def forward(self, x: torch.Tensor):
#         # x: (B, C, T, d_model)
#         B, C, T, D = x.shape

#         # get the feature
#         resident = x
#         x = x.contiguous().view(B*C, T, D)
#         x = self.attn_1(x, x, x)[0]
#         x = x.view(B, C, T, D) # (B, C, T, d_model)
#         x = self.dropout_1(x)
#         x = x + resident
#         x = self.ln_1(x)

#         x = self.activation(x) # (B, C, T, d_model)

#         # fusion the feature
#         resident = x
#         x = x.transpose(1, 2)
#         x = x.contiguous().view(-1, C, D) # (B*T, C, d_model)
#         x = self.attn_2(x, x, x)[0] # (B*T, C, d_model)
#         x = x.view(B, T, -1, D)
#         x = x.transpose(1, 2) # (B, C, T, d_model)
#         x = self.dropout_2(x)
#         x = x + resident
#         x = self.ln_2(x) # (B, C, T, d_model)

#         # feed forward
#         resident = x
#         x = x.contiguous().view(B*C, T, D)
#         x = self.ffn(x)
#         x = x.view(B, C, T, D)
#         x = self.dropout_3(x)
#         x = x + resident
#         x = self.ln_3(x)

#         return x

#     @staticmethod
#     def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
#         if activation == "relu":
#             return nn.ReLU()
#         elif activation == "gelu":
#             return nn.GELU()

#         raise RuntimeError(f"activation should be relu/gelu, not {activation}")


# class Trans_Conv(nn.Module):
#     def __init__(self, d_model: int, dropout: float, in_channels: int, out_channels: int, activation: str='gelu',
#                  **kwargs):
#         super(Trans_Conv, self).__init__(**kwargs)

#         self.attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=8,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.agg = nn.Sequential(
#             nn.Conv1d(in_channels=in_channels, out_channels=4*in_channels, kernel_size=1, bias=False),
#             self._get_activation_fn(activation),
#             nn.Dropout(dropout),
#             nn.Conv1d(in_channels=4*in_channels, out_channels=out_channels, kernel_size=1, bias=False),
#         )
        
#         self.ffn_1 = nn.Sequential(
#             nn.Linear(d_model, 4*d_model),
#             self._get_activation_fn(activation),
#             nn.Dropout(dropout),
#             nn.Linear(4*d_model, d_model),
#         )

#         self.ffn_2 = nn.Sequential(
#             nn.Linear(d_model, 4*d_model),
#             self._get_activation_fn(activation),
#             nn.Dropout(dropout),
#             nn.Linear(4*d_model, d_model),
#         )

#         self.ln_1 = nn.LayerNorm(d_model)
#         self.ln_2 = nn.LayerNorm(d_model)
#         self.ln_3 = nn.LayerNorm(d_model)
#         self.ln_4 = nn.LayerNorm(d_model)

#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)
#         self.dropout_3 = nn.Dropout(dropout)

#         self.activation = self._get_activation_fn(activation)

    
#     def forward(self, x: torch.Tensor):
#         # x: (B, C, T, d_model)
#         B, C, T, D = x.shape

#         resident = x
#         x = self.ffn_1(x)
#         x = self.dropout_1(x)
#         x = 0.5 * x + resident
#         x = self.ln_1(x)

#         # get the feature
#         resident = x
#         x = x.contiguous().view(B*C, T, D)
#         x = self.attn(x, x, x)[0]
#         x = x.view(B, C, T, D) # (B, C, T, d_model)
#         x = x + resident
#         x = self.ln_2(x)

#         # fusion the feature
#         resident = x
#         x = x.transpose(1, 2)
#         x = x.contiguous().view(-1, C, D) # (B*T, C, d_model)
#         x = self.agg(x) #
#         x = x.view(B, T, -1, D)
#         x = x.transpose(1, 2) # (B, C, T, d_model)
#         x = self.dropout_2(x)
#         x = x + resident
#         x = self.ln_3(x) # (B, C, T, d_model)

#         # feed forward
#         resident = x
#         x = self.ffn_2(x)
#         x = self.dropout_3(x)
#         x = 0.5 * x + resident
#         x = self.ln_4(x)

#         return x

#     @staticmethod
#     def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
#         if activation == "relu":
#             return nn.ReLU()
#         elif activation == "gelu":
#             return nn.GELU()

#         raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    
# class Trans_Conv(nn.Module):
#     def __init__(self, d_model: int, dropout: float, in_channels: int, out_channels: int, activation: str='gelu',
#                  **kwargs):
#         super(Trans_Conv, self).__init__(**kwargs)

#         self.attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=8,
#             dropout=dropout,
#             batch_first=True,
#         )
        
#         self.ffn_1 = nn.Sequential(
#             nn.Linear(d_model, 4*d_model),
#             self._get_activation_fn(activation),
#             nn.Dropout(dropout),
#             nn.Linear(4*d_model, d_model),
#         )

#         self.ffn_2 = nn.Sequential(
#             nn.Linear(d_model, 4*d_model),
#             self._get_activation_fn(activation),
#             nn.Dropout(dropout),
#             nn.Linear(4*d_model, d_model),
#         )

#         self.ln_1 = nn.LayerNorm(d_model)
#         self.ln_2 = nn.LayerNorm(d_model)
#         self.ln_3 = nn.LayerNorm(d_model)

#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)

#         self.activation = self._get_activation_fn(activation)

    
#     def forward(self, x: torch.Tensor):
#         # x: (B, C, T, d_model)
#         B, C, T, D = x.shape

#         resident = x
#         x = self.ffn_1(x)
#         x = self.dropout_1(x)
#         x = 0.5 * x + resident
#         x = self.ln_1(x)

#         # get the feature
#         resident = x
#         x = x.contiguous().view(B*C, T, D)
#         x = self.attn(x, x, x)[0]
#         x = x.view(B, C, T, D) # (B, C, T, d_model)
#         x = x + resident
#         x = self.ln_2(x)

#         # feed forward
#         resident = x
#         x = self.ffn_2(x)
#         x = self.dropout_2(x)
#         x = 0.5 * x + resident
#         x = self.ln_3(x)

#         return x

#     @staticmethod
#     def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
#         if activation == "relu":
#             return nn.ReLU()
#         elif activation == "gelu":
#             return nn.GELU()

#         raise RuntimeError(f"activation should be relu/gelu, not {activation}")

class Trans_Conv(nn.Module):
    def __init__(self, d_model: int, dropout: float, in_channels: int, out_channels: int, activation: str='gelu',
                 **kwargs):
        super(Trans_Conv, self).__init__(**kwargs)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

    
    def forward(self, x: torch.Tensor, padding:torch.Tensor=None):
        # x: (B, C, T, d_model)
        B, C, T, D = x.shape

        # get the feature
        x = x.contiguous().view(B*C, T, D)
        mask = torch.concat([torch.zeros(B*C, 1).cuda(), padding], dim=-1)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.view(B, C, T, D) # (B, C, T, d_model)

        return x
