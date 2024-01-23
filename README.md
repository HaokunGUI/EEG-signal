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

- BERT_231208_201200: transformer + activation + itransformer mask_ratio 0.5 dropout 0.3
BERT_231209_124620 cls linear_dropout 0.75 lr 1e-3
BERT_231209_135720 cls linear_dropout 0.3 lr 1e-4

- BERT_231209_195611: conformer (-conv1d + agg) mask_ratio 0.5 dropout 0.3
BERT_231210_125910 cls linear_dropout 0.3
BERT_231210_141609 cls linear_dropout 0.5
BERT_231210_201414 cls linear_dropout 0.6
BERT_231210_235904 cls linear_dropout 0.4

- BERT_231212_200650： conformer (-conv1d + agg) mask_ratio 0.5 dropout 0.3 agg
BERT_231213_160514 cls linear_dropout 0.4
BERT_231213_193935 cls linear_dropout 0.5
BERT_231213_154210 cls linear_dropout 0.6

BERT_231220_205032 cls linear_dropout 0.6 * 

- BERT_231214_084320: align
BERT_231214_224213 cls linear_dropout 0.6

- BERT_231215_200039: agg + align 4
BERT_231216_123945 cls linear_dropout 0.6
BERT_231220_184227 cls linear_dropout 0.6 *

- BERT_231216_152126: agg + align 1
BERT_231217_123821 cls linear_dropout 0.6
BERT_231220_194835 cls linear_dropout 0.6 *

- BERT_231218_173658: agg + align 2
BERT_231220_160342 cls linear_dropout 0.6

- BERT_231219_163430: agg + align 8
BERT_231220_171409 cls linear_dropout 0.6

- BERT_231220_215435: agg + align 1 * 
BERT_231221_134748 cls linear_dropout 0.6

- BERT_231221_214859: baseline
BERT_231222_145708 cls linear_dropout 0.6

-----------------------------------------------
1/2 FFN + MHSA + 1/2 FFN

- BERT_231226_185457: agg4 + align
BERT_231227_113814 cls linear_dropout 0.6

-----------------------------------------------
Transformer:

- BERT_231231_015811: agg2 + align
BERT_231231_124744 cls linear_dropout 0.6  0.9069

- BERT_231230_023558: agg4 
BERT_231231_124824 cls linear_dropout 0.6  0.8895

- BERT_231230_021432: baseline
BERT_231231_141224 cls linear_dropout 0.6  0.8799

- BERT_231228_234203: agg8 + align
BERT_231229_121817 cls linear_dropout 0.6  0.8627

- BERT_231228_004628: agg4 + align
BERT_231228_224110 cls linear_dropout 0.6  0.8994

- BERT_231231_144036: align
BERT_240101_013431 cls linear_dropout 0.6 
BERT_240110_214553 input_len 12

- BERT_240101_013320: agg6 + align

Ti_MAE
- ssl/Ti_MAE/Ti_MAE_231207_165555
Ti_MAE_231208_125616 cls

PatchTST
- anomaly_detection BERT_240120_020348 60
- anomaly_detection BERT_240120_130220 12
- classification BERT_240120_182616 60
- classification BERT_240120_190537 12

Ti-MAE
- lp: 
12: Ti_MAE_240121_082833
60: Ti_MAE_240121_171739

- ft:
12: Ti_MAE_240121_123255
60: Ti_MAE_240121_161420

VQ-MTM:
- lp: 
12: BERT_240120_234236
60: BERT_240120_222747