import sys
sys.path.append('/home/guihaokun/Time-Series-Pretrain')
import torch
import torch.nn as nn
from utils.utils import compute_mask_indices
from layers.BERT_Blocks import ConformerEncoderLayer
from layers.Embed import PositionalEmbedding, Tokenizer
from layers.Quantize import Quantize
import argparse

class VQ_BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, in_channel:int, patch_size:int, patch_num:int, hidden:int, n_layers: int, 
                 attn_heads:int, num_embedding: int, vq_dim: int, codebook_num: int, dropout=0.1, 
                 mask_ratio=0.2, conv_kernel_size=5, task_name='ssl', mask_length=10, no_overlap=False, 
                 min_space=1, mask_dropout=0.0, ):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(VQ_BERT, self).__init__()
        self.hidden = hidden
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
        self.num_embedding = num_embedding

        self.tokenizer = Tokenizer(in_channel=in_channel, patch_size=patch_size, 
                                   embedding_dim=hidden, shared_embedding=False)
        self.positional_encoding = PositionalEmbedding(hidden, max_len=patch_num)
        self.quantize = Quantize(
            input_dim=hidden,
            embed_dim=vq_dim,
            num_embed=num_embedding,
            codebook_num=codebook_num
            )

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4
        self.dropout = nn.Dropout(p=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [ConformerEncoderLayer(hidden, self.feed_forward_hidden, attn_heads, dropout, conv_kernel_size) 
            for _ in range(n_layers)])
        
        if task_name == 'ssl':
            self.final_projector = nn.ModuleList()
            for _ in range(codebook_num):
                self.final_projector.append(nn.Linear(hidden, num_embedding))
        elif task_name == 'anomaly_detection':
            self.final_projector = nn.Linear(hidden*patch_num*in_channel, 1)


    def forward(self, x):
        # attention masking for padded token
        # x:[batch_size, seq_len, in_channel]

        # tokenize the input 
        token = self.tokenizer(x) # [batchsize, nvar, patch_num, embedding_dim]
        B, N, T, D = token.shape
        # quantize the input
        quant_idx = self.quantize(token) # [batchsize, nvar, patch_num, codebook_num]
        # create random masking
        mask = compute_mask_indices((B*N, T), 
                                    None,
                                    mask_prob=self.mask_ratio, 
                                    mask_length=self.mask_length, 
                                    no_overlap=self.no_overlap,
                                    min_space=self.min_space,
                                    mask_dropout=self.mask_dropout,
                                    ) #[bs*nvar, T]
        mask = torch.from_numpy(mask).to(x.device)
        token = token.view(B*N, T, D) #[bs*nvar, T, D]
        masked_num = mask.sum() #[bs*nvar]
        random_sample = torch.normal(mean=0, std=0.1, size=(masked_num, D)).to(x.device) #[bs*nvar, masked_num, D]
        masked_token = token.clone()
        masked_token[mask] = random_sample
        masked_token = masked_token.view(B, N, T, D)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            xm = transformer.forward(masked_token, None) # [batchsize, nvar, patch_num, embedding_dim]
        
        if self.task_name == 'ssl':
            mask_xm = xm.view(-1, *xm.shape[2:])[mask] #[masked_num, embedding_dim]
            possibility = []
            for i in range(self.codebook_num):
                possibility.append(self.final_projector[i](mask_xm).unsqueeze(-2)) # [masked_num, ?, num_embedding]
            possibility = torch.cat(possibility, dim=-2) # [masked_num, codebook_num, num_embedding]
            quant_mask = quant_idx.view(-1, *quant_idx.shape[2:])[mask] # [masked_num, codebook_num]
            return possibility.view(-1, self.num_embedding), quant_mask.view(-1)
        elif self.task_name == 'anomaly_detection':
            flatten = nn.Flatten(start_dim=-2)
            xm = flatten(xm)
            xm = self.dropout(xm)
            xm = xm.view(xm.shape[0], -1)
            xm = self.final_projector(xm)
            return xm
        else:
            return xm

class Model(nn.Module):
    def __init__(self, args:argparse.Namespace):
        super(Model, self).__init__()
        if args.task_name == 'ssl':
            patch_num = args.input_len + args.output_len
        else:
            patch_num = args.input_len
        self.model = VQ_BERT(
            in_channel=args.num_nodes,
            patch_size=args.freq,
            patch_num=patch_num,
            hidden=args.d_model,
            n_layers=args.e_layers,
            attn_heads=args.attn_head,
            num_embedding=args.num_embedding,
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
        )

    def forward(self, x:torch.Tensor):
        # BCT -> BTC
        x = x.permute(0, 2, 1)
        return self.model.forward(x)
