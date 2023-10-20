import torch
import torch.nn as nn
from utils.utils import random_masking
from layers.BERT_Blocks import TransformerBlock
from layers.Embed import PositionalEmbedding, Tokenizer
from layers.Quantize import Quantize
import argparse

class VQ_BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, in_channel:int, patch_size:int, patch_num:int, hidden:int, embedding_dim: int, 
                 n_layers: int, attn_heads:int, num_embedding: int, dropout=0.1, mask_ratio=0.2):
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
        self.num_embedding = num_embedding
        self.mask_ratio = mask_ratio

        self.tokenizer = Tokenizer(in_channel=in_channel, patch_size=patch_size, embedding_dim=embedding_dim, shared_embedding=False)
        self.positional_encoding = PositionalEmbedding(embedding_dim, max_len=patch_num)
        self.quantize = Quantize(num_embeddings=num_embedding, embedding_dim=embedding_dim, embedding_method='fixed')

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4
        self.dropout = nn.Dropout(p=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # x:[batch_size, seq_len, in_channel]

        # tokenize the input 
        token = self.tokenizer(x) # [batchsize, nvar, patch_num, embedding_dim]
        B, N, P, D = token.shape
        # quantize the input
        token = self.quantize(token.view(-1, P, D)) # [batchsize*nvar, patch_num, embedding_dim]
        # add positional encoding to the input
        token = self.dropout(token + self.positional_encoding(token)).view(B, N, P, D) # [batchsize, nvar, patch_num, embedding_dim]

        # create random masking
        xm, _, mask, _ = random_masking(token, self.mask_ratio) # [batchsize, nvar, patch_num, embedding_dim]
        xm = xm.view(-1, P, D) # [batchsize*nvar, patch_num, embedding_dim]

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            xm = transformer.forward(xm)
        xm = xm.view(B, N, P, D) # [batchsize, nvar, patch_num, embedding_dim]

        return xm, mask
    
class Model(nn.Moduke):
    def __init__(self, args:argparse.Namespace):
        super(Model, self).__init__()
        pass

    def forward(self, x:torch.Tensor):
        pass
