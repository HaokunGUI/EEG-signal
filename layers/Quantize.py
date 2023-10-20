import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantize(nn.Module):
    '''
    A PyTorch module for vector quantization.

    Args:
        num_embeddings (int): The number of embedding vectors (codebook entries).
        embedding_dim (int): The dimension of each embedding vector.
        embedding_method (str): Specifies the embedding method, either 'fixed' or 'learned'.
        decay (int): A decay factor for updating learned embeddings (only used when embedding_method is 'learned').
        eps (int): A small constant to prevent division by zero (only used when embedding_method is 'learned').
    '''
    def __init__(self, num_embeddings: int, embedding_dim: int, embedding_method: str = 'fixed', 
                 decay: int = 0.99, eps: int = 1e-5):
        super(Quantize, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding_method = embedding_method

        # Initialize the embedding vectors based on the chosen method.
        if embedding_method == 'fixed':
            embed = nn.Embedding(self.num_embeddings, self.embedding_dim)
            nn.init.xavier_normal_(embed.weight)
            embed.weight.requires_grad = False
        elif embedding_method == 'learned':
            self.decay = decay
            self.eps = eps
            embed = torch.randn(embedding_dim, num_embeddings)
            self.register_buffer('cluster_size', torch.zeros(num_embeddings))
            self.register_buffer("embed_avg", embed.clone())
        else:
            raise ValueError('Unknown embedding method: ' + embedding_method)

        self.register_buffer('embed', embed)
        
    def forward(self, input: torch.Tensor):
        # Flatten the input tensor and compute distances to the embedding vectors.
        flatten = input.reshape(-1, self.embedding_dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        # Update the embedding vectors if in training mode and using 'learned' method.
        if self.training and self.embedding_method == 'learned':
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        
        # Compute the difference between the quantized and input vectors.
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind
    
    def embed_code(self, embed_id: torch.Tensor):
        # Retrieve the embeddings for given indices.
        return F.embedding(embed_id, self.embed.transpose(0, 1))