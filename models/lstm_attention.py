"""
LSTM decoder with attention, adapted from the Stanza toolkit.
"""

import torch
from torch import nn
from stanza.models.common.seq2seq_modules import BasicAttention, LinearAttention, DeepAttention

class LSTMAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, nlayers, batch_first=True, attn_type='soft', dropout=0, hidden_size2=0, pair_level_attn=False):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.lstm_cell = nn.LSTM(input_size, hidden_size, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.attn_type = attn_type

        if pair_level_attn:
            self.comb = nn.Linear(hidden_size, hidden_size)

        if attn_type == 'soft':
            self.attention_layer = SoftDotAttention(hidden_size)
        elif attn_type == 'mlp':
            self.attention_layer = BasicAttention(hidden_size)
        elif attn_type == 'linear':
            self.attention_layer = LinearAttention(hidden_size)
        elif attn_type == 'deep':
            self.attention_layer = DeepAttention(hidden_size)
        elif attn_type == 'none':
            pass
        else:
            raise Exception("Unsupported LSTM attention type: {}".format(attn_type))
        print("Using {} attention for LSTM.".format(attn_type))

    def forward(self, input, hidden, ctx, ctx_mask=None, h_bg=None, pair_level_output=None, tok_level_attn=None, turn_ids=None, previous_output=None):
        """Propogate input through the network."""
        if self.batch_first:
            input = input.transpose(0,1)

        output = []
        alphas = []
        steps = range(input.size(0))
        hid_size = hidden[0].size()
        hid0 = hidden[0][-1]
        hidden = tuple(x.view(1, x.size(0) * x.size(1), *x.size()[2:]) for x in hidden)
        if h_bg is not None:
            h_bg = self.drop(h_bg)
        for i in steps:
            step_input = input[i]
            if h_bg is not None:
                step_input = torch.cat([step_input, h_bg], -1)
            hy, hidden = self.lstm_cell(step_input.unsqueeze(1), hidden)
            hy = hy.squeeze(1)
            if self.attn_type == 'none':
                h_tilde = hy
                alpha = hy.new_zeros((1,))
            else:
                h_tilde, alpha = self.attention_layer(self.drop(hy), self.drop(ctx), mask=ctx_mask, tok_level_attn=tok_level_attn)

            if pair_level_output is not None:
                h_tilde = torch.tanh(self.comb(torch.cat([h_tilde, pair_level_output], -1)))

            output.append(h_tilde)
            alphas.append(alpha)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        alphas = torch.stack(alphas)

        hidden = tuple(x.view(hid_size) for x in hidden)

        if self.batch_first:
            output = output.transpose(0,1)
            alphas = alphas.transpose(0,1)

        return output, hidden, alphas

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim, dim2=None, heads=1):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        dim2 = dim2 if dim2 is not None else dim
        self.linear_in = nn.Linear(dim, dim2)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim + dim2, dim)
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False, tok_level_attn=None, key=None):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
        key = context if key is None else key

        # Get attention

        attn = torch.bmm(key, target).squeeze(2)  # batch x sourceL

        if tok_level_attn is not None:
            attn += torch.log(tok_level_attn + 1e-12)

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -1e12)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim

        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = torch.tanh(self.linear_out(h_tilde))

        return h_tilde, attn

