# organized order: quantizer & tokenizer in 2 branches.
- ssl/VQ_BERT/VQ_BERT_231113_132821/log: no normalization
- ssl/VQ_BERT/VQ_BERT_231115_082946/log: normalize before quantizer
- ssl/VQ_BERT/BERT_with_resident/log: using resident after input
- ssl/VQ_BERT/VQ_BERT_231116_175206/log: using IN after input

# organized order: quantizer followed by tokenizer
- ssl/VQ_BERT/VQ_BERT_231121_205431: tokenizer + LN/ input - IN
VQ_BERT_231122_151412 fix & max
VQ_BERT_231122_175339 fix & random init & max
