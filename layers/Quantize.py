import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantize(nn.Module):
    '''
    A PyTorch module for vector quantization.

    Args: input_dim: int, input dimension
            embed_dim: int, embedding dimension
            num_embed: int, number of embedding vectors
            codebook_num: int, number of codebooks
    '''
    def __init__(self, input_dim: int, vq_dim: int, num_embed: int, codebook_num: int):
        super(Quantize, self).__init__()
        projector = torch.nn.init.xavier_normal_(torch.empty(input_dim, vq_dim))
        projector.requires_grad = False
        self.register_buffer('projector', projector)
        codebook = torch.nn.init.normal_(torch.empty(codebook_num, num_embed, vq_dim))
        codebook.requires_grad = False
        self.register_buffer('codebook', codebook)
        
    def forward(self, input: torch.Tensor):
        """
        :param input: [bs, T, input_dim]
        """
        x = input.matmul(self.projector)
        x_norm = x.norm(dim=-1, keepdim=True)
        x = x / x_norm # [bs, T, embed_dim]
        codebook_norm = self.codebook.norm(dim=-1, keepdim=True)
        codebook = self.codebook / codebook_norm # [codebook_num, num_embed, embed_dim]
        
        similarity = torch.einsum('btd,cmd->btcm', x, codebook)
        idx = torch.argmax(similarity, dim=-1)
        return idx
