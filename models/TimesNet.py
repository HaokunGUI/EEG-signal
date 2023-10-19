import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, args):
        super(TimesBlock, self).__init__()
        self.input_len = args.input_dim
        if args.task_name == 'anomaly_detection':
            self.output_len = 0
        else:
            self.output_len = args.output_dim
        self.k = args.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(args.d_model, args.d_hidden,
                               num_kernels=args.num_kernels),
            nn.GELU(),
            Inception_Block_V1(args.d_hidden, args.d_model,
                               num_kernels=args.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.input_len + self.output_len) % period != 0:
                length = (((self.input_len + self.output_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.input_len + self.output_len)), x.shape[2]]).cuda()
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.input_len + self.output_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.input_len + self.output_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.task_name = args.task_name
        self.input_len = args.input_dim
        self.output_len = args.output_dim
        self.model = nn.ModuleList([TimesBlock(args)
                                    for _ in range(args.e_layers)])
        self.enc_embedding = DataEmbedding(args.num_nodes, args.d_model, args.input_dim, args.dropout)
        self.layer = args.e_layers
        self.layer_norm = nn.LayerNorm(args.d_model)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                args.d_model, args.num_nodes, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(args.dropout)
            self.projection = nn.Linear(
                args.d_model * args.input_len, args.num_class)

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.input_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.input_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc):
        x_enc = x_enc.permute(0, 2, 1)
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            dec_out = dec_out.permute(0, 2, 1)
            return dec_out  # [B, T, C]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            dec_out = dec_out.permute(0, 2, 1)
            return dec_out  # [B, N]
        return None
