import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils.utils import compute_mask_indices
from typing import Callable
from layers.BERT_Enc import Trans_Conv
from layers.Quantize import Quantize
from layers.Embed import PositionalEmbedding

class BERT(nn.Module):
    def __init__(self, d_model: int, patch_size: int, dropout: float, in_channels: int, hidden_channels: int, mask_ratio: float,
                 num_layers: int, codebook_size: int, activation: str='gelu', task_name: str='ssl', linear_dropout: float=0.5,
                 mask_type: str='poisson', **kwargs):
        
        super(BERT, self).__init__(**kwargs)

        # Hyper-parameters
        self.patch_size = patch_size
        self.task_name = task_name
        self.hidden_channels = hidden_channels
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.d_model = d_model

        # Project the patch_dim to the d_model dimension
        self.embed = nn.Linear(patch_size, d_model)
        self.activation_embed = self._get_activation_fn(activation)
        self.norm = nn.LayerNorm(d_model)
        
        # embed the time series to patch
        self.pos_embed = PositionalEmbedding(
            d_model=d_model,
            max_len=100
        )

        # Aggregation
        self.agg = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1
        )
        
        # Transformer Encoder
        self.TOKEN_CLS = torch.normal(mean=0, std=0.1, size=(1, 1, 1, d_model)).cuda()
        self.TOKEN_CLS.requires_grad = False
        self.register_buffer("CLS", self.TOKEN_CLS)
        
        self.encoder = nn.ModuleList([
            Trans_Conv(
                d_model=d_model, 
                dropout=dropout, 
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # Different decoder for different down stream tasks
        if task_name == 'ssl':
            self.quantizer = Quantize(
                input_dim=patch_size,
                vq_dim=d_model,
                num_embed=codebook_size,
                codebook_num=1
            )

            self.decoder = nn.Conv1d(in_channels=hidden_channels,
                                     out_channels=in_channels,
                                     kernel_size=1)
            self.activation = self._get_activation_fn(activation)
            self.final_projector = nn.Linear(d_model, codebook_size)
        elif task_name == 'anomaly_detection':
            self.decoder_ad = nn.Conv1d(in_channels=hidden_channels,
                                     out_channels=1,
                                     kernel_size=1)
            self.activation = self._get_activation_fn(activation)
            self.final_projector_ad = nn.Linear(d_model, 1)
        else:
            raise RuntimeError(f"task_name should be ssl/anomaly_detection, not {task_name}")
        
        self.linear_dropout = nn.Dropout(p=linear_dropout)
        

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        B, C, T = x.shape
        assert T % self.patch_size == 0, f"Time series length should be divisible by patch_size, not {T} % {self.patch_size}"
        x = x.view(B, C, -1, self.patch_size) # (B, C, T, patch_size)

        # Embedding
        y = self.embed(x) # (B, C, T, d_model)
        y = self.activation_embed(y)
        y = self.norm(y) # (B, C, T, d_model)

        # Add CLS token
        y = torch.concat([self.CLS.repeat(B, C, 1, 1), y], dim=2) # (B, C, T+1, d_model) 

        # Random masking
        if self.task_name == 'ssl':
            mask = self.random_masking(
                shape=(B*C, T // self.patch_size),
                mask_ratio=self.mask_ratio,
                device=x.device,
            )
            masked_num = mask.sum() #[bs]
            random_sample = torch.normal(mean=0, std=0.1, size=(masked_num, self.d_model)).to(x.device) #[bs, masked_num, D]
            y = y.view(B*C, *y.shape[2:]) # (B*C, T+1, d_model)
            y[:, 1:, :][mask] = random_sample
            y = y.view(B, C, *y.shape[1:]) # (B, C, T+1, d_model)

        y = y.transpose(1, 2)
        y = y.contiguous().view(-1, C, y.shape[-1]) # (B*(T+1), C, d_model)
        y = self.agg(y) # (B*(T+1), C', d_model)
        y = y.view(B, -1, self.hidden_channels, y.shape[-1]) # (B, T+1, C', d_model)
        y = y.transpose(1, 2) # (B, C', T+1, d_model)
        y = y.contiguous().view(-1, *y.shape[2:]) # (B*C', T+1, d_model)
        pos_embed = self.pos_embed(y) # (B*C', T+1, d_model)
        y = y + pos_embed

        # Transformer Encoder
        y = y.view(B, self.hidden_channels, *y.shape[1:]) # (B, C', T+1, d_model)
        for encoder in self.encoder:
            y = encoder(y) # (B, C', T+1, d_model)
        
        # Decoder
        if self.task_name == 'ssl':
            # Get the idx
            idx = self.quantizer(x.view(-1, *x.shape[2:])) # (B*C, T, 1)
            idx = idx[mask].squeeze(-1) # (masked_num)

            # Get the prediction
            y = y.transpose(1, 2) # (B, T+1, C', d_model)
            y = y.contiguous().view(-1, *y.shape[2:]) # (B*(T+1), C', d_model)
            y = self.decoder(y) # (B(*(T+1), C, d_model)
            y = y.view(B, -1, *y.shape[1:]) # (B, T+1, C, d_model)
            y = y.transpose(1, 2) # (B, C, T+1, d_model)
            y = self.activation(y) # (B, C, T+1, d_model)
            y = y[:, :, 1:, :] # (B, C, T, d_model)
            y = y.contiguous().view(-1, *y.shape[2:]) # (B*C, T, d_model)
            y = y[mask] # (masked_num, d_model)
            y = self.linear_dropout(y)
            y = self.final_projector(y) # (masked_num, codebook_size)

            return y, idx
       
        elif self.task_name == 'anomaly_detection':
            y = torch.mean(y[:, :, 1:, :], dim=2).squeeze(2) # (B, C', d_model)
            y = self.decoder_ad(y).squeeze(1) # (B, d_model)
            y = self.activation(y) # (B, d_model)
            y = self.linear_dropout(y)
            y = self.final_projector_ad(y) # (B, 1)
            return y
        else:
            raise RuntimeError(f"task_name should be ssl/anomaly_detection, not {self.task_name}")

    @staticmethod
    def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    
    def random_masking(self, shape, mask_ratio, device):
        N, L = shape  # batch, length
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # create mask
        mask = torch.ones([N, L], dtype=bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :len_keep], 0)

        return mask
    

class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Model, self).__init__()
        self.args = args
        self.model = BERT(
            d_model=args.d_model,
            patch_size=args.freq,
            dropout=args.dropout,
            in_channels=args.num_nodes,
            hidden_channels=args.hidden_channels,
            num_layers=args.e_layers,
            codebook_size=args.codebook_item,
            activation=args.activation,
            task_name=args.task_name,
            linear_dropout=args.linear_dropout,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
        )
    
    def forward(self, x: torch.Tensor):
        return self.model.forward(x)
