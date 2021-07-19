import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NARM(nn.Module):
    def __init__(self, args):
        super(NARM, self).__init__()
        self.args = args
        self.embedding = NARMEmbedding(self.args)
        self.model = NARMModel(self.args)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for p in self.parameters():
                p.uniform_(2 * l - 1, 2 * u - 1)
                p.erfinv_()
                p.mul_(std * math.sqrt(2.))
                p.add_(mean)

    def forward(self, x, lengths):
        x, mask = self.embedding(x, lengths)
        scores = self.model(x, self.embedding.token.weight, lengths, mask)
        return scores


class NARMEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 1
        embed_size = args.bert_hidden_units
        
        self.token = nn.Embedding(vocab_size, embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)

    def get_mask(self, x, lengths):
        if len(x.shape) > 2:
            return torch.ones(x.shape[:2])[:, :max(lengths)].to(x.device)
        else:
            return ((x > 0) * 1)[:, :max(lengths)]

    def forward(self, x, lengths):
        mask = self.get_mask(x, lengths)
        if len(x.shape) > 2:
            x = torch.matmul(x, self.token.weight)
        else:
            x = self.token(x)

        return self.embed_dropout(x), mask


class NARMModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        embed_size = args.bert_hidden_units
        hidden_size = 2 * args.bert_hidden_units

        self.gru = nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.a_global = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a_local = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = HardSigmoid()
        self.v_vector = nn.Linear(hidden_size, 1, bias=False)
        self.proj_dropout = nn.Dropout(args.bert_attn_dropout)
        self.b_vetor = nn.Linear(embed_size, 2 * hidden_size, bias=False)

    def forward(self, x, embedding_weight, lengths, mask):
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(x)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
        c_global = hidden[-1]

        state2 = self.a_local(gru_out)
        state1 = self.a_global(c_global).unsqueeze(1).expand_as(state2)
        state1 = mask.unsqueeze(2).expand_as(state2) * state1
        alpha = self.act(state1 + state2).view(-1, state1.size(-1))
        attn = self.v_vector(alpha).view(mask.size())
        attn = F.softmax(attn.masked_fill(mask == 0, -1e9), dim=-1)
        c_local = torch.sum(attn.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        proj = self.proj_dropout(torch.cat([c_global, c_local], 1))
        scores = torch.matmul(proj, self.b_vetor(embedding_weight).permute(1, 0))
        return scores


class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp((x / 6 + 0.5), min=0., max=1.)