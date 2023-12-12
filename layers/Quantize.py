import torch
import torch.nn as nn
import torch.nn.functional as F

# class Quantize(nn.Module):
#     '''
#     A PyTorch module for vector quantization.

#     Args: input_dim: int, input dimension
#             embed_dim: int, embedding dimension
#             num_embed: int, number of embedding vectors
#             codebook_num: int, number of codebooks
#     '''
#     def __init__(self, input_dim: int, vq_dim: int, num_embed: int, codebook_num: int):
#         super(Quantize, self).__init__()
#         projector = torch.nn.init.xavier_normal_(torch.empty(input_dim, vq_dim))
#         projector.requires_grad = False
#         self.register_buffer('projector', projector)
#         codebook = torch.nn.init.normal_(torch.empty(codebook_num, num_embed, vq_dim))
#         codebook.requires_grad = False
#         self.register_buffer('codebook', codebook)
        
#     def forward(self, input: torch.Tensor):
#         """
#         :param input: [bs, T, input_dim]
#         """
#         x = input.matmul(self.projector)
#         x_norm = x.norm(dim=-1, keepdim=True)
#         x = x / x_norm # [bs, T, embed_dim]
#         codebook_norm = self.codebook.norm(dim=-1, keepdim=True)
#         codebook = self.codebook / codebook_norm # [codebook_num, num_embed, embed_dim]
        
#         similarity = torch.einsum('btd,cmd->btcm', x, codebook)
#         idx = torch.argmax(similarity, dim=-1)
#         return idx


class Quantize(nn.Module):
    def __init__(self, input_dim: int, vq_dim: int, num_embed: int, codebook_num: int=1, split_num:int=4, **kwargs):
        super(Quantize, self).__init__(**kwargs)
        # hyper-parameters
        self.split_num = split_num

        self.projector = torch.nn.init.xavier_normal_(torch.empty(input_dim, vq_dim))
        self.projector = nn.Parameter(self.projector, requires_grad=False)

        codebook = torch.nn.init.normal_(torch.empty(codebook_num, num_embed, vq_dim))
        self.codebook = nn.Parameter(codebook, requires_grad=False)

        self.random_matrix = torch.randint(0, split_num, (input_dim//2 + 1, ))
        self.random_matrix = nn.Parameter(self.random_matrix, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [bs, T, input_dim]
        with torch.no_grad():
            x_fft = torch.fft.rfft(x, dim=-1)
            x_fft_subsample = [x_fft * (self.random_matrix == i) for i in range(self.split_num)]
            x_fft_subsample = torch.stack(x_fft_subsample, dim=2) # [bs, T, split_num, input_dim]
            x_recon = torch.fft.irfft(x_fft_subsample, dim=-1) # [bs, T, split_num, input_dim]

            x_feature = x_recon.matmul(self.projector) # [bs, T, split_num, vq_dim]

            x_feature_norm = x_feature.norm(dim=-1, keepdim=True)
            x_feature = x_feature / x_feature_norm # [bs, T, split_num, vq_dim]
            codebook_norm = self.codebook.norm(dim=-1, keepdim=True)
            codebook = self.codebook / codebook_norm # [codebook_num, num_embed, vq_dim]

            similarity = torch.einsum('btnd,cmd->btncm', x_feature, codebook) # [bs, T, split_num, codebook_num, num_embed]
            idx = torch.argmax(similarity, dim=-1) # [bs, T, split_num, codebook_num]
            return idx
