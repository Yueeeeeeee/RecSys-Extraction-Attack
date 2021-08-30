import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.pe = nn.Embedding(max_len+1, d_model)

    def forward(self, x):
        pose = (x > 0) * (x > 0).sum(dim=-1).unsqueeze(1).repeat(1, x.size(-1))
        pose += torch.arange(start=-(x.size(1)-1), end=1, step=1, device=x.device)
        pose = pose * (x > 0)

        return self.pe(pose)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))


# layer norm
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


# layer norm and dropout (dropout and then layer norm)
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))  # original implementation
        return self.layer_norm(x + self.dropout(sublayer(x)))  # BERT4Rec implementation


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None, sas=False):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if sas:
            direction_mask = torch.ones_like(scores)
            direction_mask = torch.tril(direction_mask)
            scores = scores.masked_fill(direction_mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, head_size=None, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.h = h
        self.d_k = d_model // h
        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, self.h * self.head_size) for _ in range(3)])
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(self.h * self.head_size, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        # 2) apply attention on all the projected vectors in batch.
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # 3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.head_size)
        return self.output_linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class SASMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, head_size=None, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.h = h
        self.d_k = d_model // h
        if head_size is not None:
            self.head_size = head_size
        else:
            self.head_size = d_model // h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, self.h * self.head_size) for _ in range(3)])
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) do all the linear projections in batch from d_model => h x d_k
        query_, key_, value_ = [l(x).view(batch_size, -1, self.h, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        # 2) apply attention on all the projected vectors in batch.
        x, attn = self.attention(
            query_, key_, value_, mask=mask, dropout=self.dropout, sas=True)

        # 3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.head_size)
        
        return self.layer_norm(x + query)


class SASPositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        return self.layer_norm(self.dropout(self.conv2(x_)).permute(0, 2, 1) + x)


class SASTransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, head_size, feed_forward_hidden, dropout, attn_dropout=0.1):
        super().__init__()
        self.layer_norm = LayerNorm(hidden)
        self.attention = SASMultiHeadedAttention(
            h=attn_heads, d_model=hidden, head_size=head_size, dropout=attn_dropout)
        self.feed_forward = SASPositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

    def forward(self, x, mask):
        x = self.attention(self.layer_norm(x), x, x, mask)
        x = self.feed_forward(x)
        return x