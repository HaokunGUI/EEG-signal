# organized order: quantizer & tokenizer in 2 branches.
- ssl/VQ_BERT/VQ_BERT_231113_132821/log: no normalization
- ssl/VQ_BERT/VQ_BERT_231115_082946/log: normalize before quantizer
- ssl/VQ_BERT/BERT_with_resident/log: using resident after input
- ssl/VQ_BERT/VQ_BERT_231116_175206/log: using IN after input
VQ_BERT_231122_222835 concat(hidden_dim 5)
VQ_BERT_231123_012549 concat(hidden_dim 1)
VQ_BERT_231123_151604 softmax sum
VQ_BERT_231123_163027 glu max
VQ_BERT_231123_185614 mean
VQ_BERT_231123_200443 max
VQ_BERT_231123_224749 temp 3 ((1 - nn.Sigmoid()(lamda * torch.max(x, dim=-1, keepdim=True)[0])).detach())
VQ_BERT_231124_002228 temp 5
VQ_BERT_231124_012626 temp 1
VQ_BERT_231124_123819 temp 5 lamda * (1 - nn.Sigmoid()(torch.max(x, dim=-1, keepdim=True)[0])).detach()


# organized order: quantizer followed by tokenizer
- ssl/VQ_BERT/VQ_BERT_231121_205431: tokenizer + LN/ input - IN
VQ_BERT_231122_151412 fix & max
VQ_BERT_231122_175339 fix & random init & max

BERT
- ssl/BERT/BERT_231201_012705: transformer + conv1d
BERT_231201_141353 cls
BERT_231201_152935 max
BERT_231201_163636 mean

- BERT_231205_224117: transformer + itranformer 0.75
BERT_231208_141332 cls

- BERT_231207_171122: transformer + itransformer 0.3
BERT_231208_114750 cls