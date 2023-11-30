import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class Trans_Conv(nn.Module):
    def __init__(self, d_model: int, dropout: float, in_channels: int, out_channels: int, activation: str='gelu'):
        super(Trans_Conv, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=1
                            )
        self.activation = self._get_activation_fn(activation)
    
    def forward(self, x: torch.Tensor):
        # x: (B, C, T, d_model)
        B, C, T, D = x.shape
        x = x.reshape(B*C, T, D) # (B*C, T, d_model)
        x = self.transformer(x) # (B*C, T, d_model)
        x = x.reshape(B, C, T, D) # (B, C, T, d_model)
        y = x.transpose(1, 2).reshape(-1, C, D) # (B*T, C, d_model)
        y = self.conv(y) # (B*T, C', d_model)
        y = y.reshape(B, T, -1, D).transpose(1, 2) # (B, C', T, d_model)
        y = self.activation(y) # (B, C', T, d_model)
        return y
    
    @staticmethod
    def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    
