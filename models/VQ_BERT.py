import sys
sys.path.append('/home/guihaokun/Time-Series-Pretrain')
import torch
import torch.nn as nn
from utils.utils import compute_mask_indices
from layers.BERT_Blocks import ConformerEncoderLayer
from layers.Embed import PositionalEmbedding, Tokenizer, RelPositionalEncoding
from layers.Quantize import Quantize
from layers.Normalize import RevIN
import argparse

class VQ_BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, in_channel:int, patch_size:int, patch_num:int, d_model:int, n_layers: int, 
                 attn_heads:int, codebook_item: int, vq_dim: int, codebook_num: int, dropout=0.1, 
                 mask_ratio=0.2, conv_kernel_size=5, task_name='ssl', mask_length=10, no_overlap=False, 
                 min_space=1, mask_dropout=0.0, enc_type='rel', mask_type='static'):
        """
        :param vocab_size: vocab_size of total words
        :param d_model: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(VQ_BERT, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.mask_ratio = mask_ratio
        self.conv_kernel_size = conv_kernel_size
        self.task_name = task_name
        self.mask_length = mask_length
        self.no_overlap = no_overlap
        self.min_space = min_space
        self.mask_dropout = mask_dropout
        self.vq_dim = vq_dim
        self.codebook_num = codebook_num
        self.codebook_item = codebook_item
        self.enc_type = enc_type
        self.mask_type = mask_type
        self.RevIN = RevIN(d_model)

        self.tokenizer = Tokenizer(in_channel=in_channel, patch_size=patch_size, 
                                   embedding_dim=d_model)
        if self.enc_type == 'abs':
            self.positional_encoding = PositionalEmbedding(d_model, max_len=patch_num)
        elif self.enc_type == 'rel':
            self.positional_encoding = RelPositionalEncoding(d_model=d_model, max_len=patch_num)
        else:
            raise ValueError('unknown positional encoding type: {}'.format(self.enc_type))
        
        self.quantize = Quantize(
            input_dim=d_model,
            vq_dim=vq_dim,
            num_embed=codebook_item,
            codebook_num=codebook_num
            )

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4
        self.dropout = nn.Dropout(p=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [ConformerEncoderLayer(d_model, self.feed_forward_hidden, attn_heads, dropout, conv_kernel_size, enc_type=enc_type) 
            for _ in range(n_layers)])
        
        if task_name == 'ssl':
            self.final_projector = nn.ModuleList()
            for _ in range(codebook_num):
                self.final_projector.append(nn.Linear(d_model, codebook_item))
        elif task_name == 'anomaly_detection':
            self.final_projector = nn.Linear(d_model*patch_num, 1)


    def forward(self, x):
        # attention masking for padded token
        # x:[batch_size, seq_len, in_channel]

        # tokenize the input 
        token = self.tokenizer(x) # [batchsize, patch_num, embedding_dim]
        token = self.RevIN(token, mode='norm')
        B, T, D = token.shape

        if self.task_name == 'ssl':
            # quantize the input
            quant_idx = self.quantize(token) # [batchsize, patch_num, codebook_num]
            
            mask = compute_mask_indices((B, T),
                                None,
                                mask_prob=self.mask_ratio, 
                                mask_length=self.mask_length, 
                                no_overlap=self.no_overlap,
                                min_space=self.min_space,
                                mask_type=self.mask_type,
                                mask_dropout=self.mask_dropout,
                            ) #[bs, T]
            mask = torch.from_numpy(mask).to(x.device)
            masked_num = mask.sum() #[bs]
            random_sample = torch.normal(mean=0, std=0.1, size=(masked_num, D)).to(x.device) #[bs, masked_num, D]
            token[mask] = random_sample

        if self.enc_type == 'abs':
            # add position encoding
            token = token + self.positional_encoding(token) # [batchsize, patch_num, embedding_dim]
            # running over multiple transformer blocks
            for transformer in self.transformer_blocks:
                xm = transformer.forward(token, None, None) # [batchsize, patch_num, embedding_dim]
        elif self.enc_type == 'rel':
            for transformer in self.transformer_blocks:
                xm = transformer.forward(token, None, self.positional_encoding(token))
        
        xm = self.RevIN(xm, mode='denorm')
        
        if self.task_name == 'ssl':
            mask_xm = xm[mask] #[masked_num, embedding_dim]
            possibility = []
            for i in range(self.codebook_num):
                possibility.append(self.final_projector[i](mask_xm).unsqueeze(-2)) # [masked_num, ?, codebook_item]
            possibility = torch.cat(possibility, dim=-2) # [masked_num, codebook_num, codebook_item]
            quant_mask = quant_idx[mask] # [masked_num, codebook_num]
            return possibility.view(-1, self.codebook_item), quant_mask.view(-1)
        elif self.task_name == 'anomaly_detection':
            xm = xm.reshape(B, -1)
            xm = self.dropout(xm)
            xm = self.final_projector(xm)
            return xm
        else:
            return xm

class Model(nn.Module):
    def __init__(self, args:argparse.Namespace):
        super(Model, self).__init__()
        self.model = VQ_BERT(
            in_channel=args.num_nodes,
            patch_size=args.freq,
            patch_num=args.input_len,
            d_model=args.d_model,
            n_layers=args.e_layers,
            attn_heads=args.attn_head,
            codebook_item=args.codebook_item,
            vq_dim=args.d_hidden,
            codebook_num=args.codebook_num,
            dropout=args.dropout,
            mask_ratio=args.mask_ratio,
            conv_kernel_size=args.kernel_size,
            task_name=args.task_name,
            mask_length=args.mask_length,
            no_overlap=args.no_overlap,
            min_space=args.min_space,
            mask_dropout=args.mask_dropout,
            enc_type=args.enc_type,
            mask_type=args.mask_type,
        )

    def forward(self, x:torch.Tensor):
        # BCT -> BTC
        x = x.permute(0, 2, 1)
        return self.model.forward(x)
