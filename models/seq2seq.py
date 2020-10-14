"""
This file contains the description of the question generation and teacher QA/classifier models, as well as helper functions.
"""

from collections import Counter
import numpy as np
import random
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.tools import squad_eval

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common import utils as seq2sequtils
from stanza.models.common.beam import Beam

from models.lstm_attention import LSTMAttention
from utils.constants import *
from models.data import MAX_SKETCHES

f1_cache = dict()
def cached_f1(pred, gold):
    if (pred, gold) not in f1_cache:
        f1_cache[(pred, gold)] = squad_eval.f1_score(pred, gold)
    return f1_cache[(pred, gold)]

prec_cache = dict()
def cached_prec(pred, gold):
    if (pred, gold) not in prec_cache:
        prediction_tokens = squad_eval.normalize_answer(pred).split()
        ground_truth_tokens = squad_eval.normalize_answer(gold).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            prec_cache[(pred, gold)] = 0
        else:
            prec_cache[(pred, gold)] = 1.0 * num_same / len(prediction_tokens)

    return prec_cache[(pred, gold)]

def zero_state(inputs, encoder, use_cuda=True):
    batch_size = inputs.size(0)
    h0 = torch.zeros(encoder.num_layers*(2 if encoder.bidirectional else 1), batch_size, encoder.hidden_size, requires_grad=False)
    c0 = torch.zeros(encoder.num_layers*(2 if encoder.bidirectional else 1), batch_size, encoder.hidden_size, requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    return h0, c0

def keep_partial_grad(grad, topk=None):
    """
    Keep only the topk rows of grads.
    """
    assert topk is None or topk < grad.size(0)
    if topk is not None:
        grad.data[topk:].zero_()
    grad.data[0].zero_()
    return grad

def encode(enc_inputs, lens, encoder, h0=None, c0=None, use_cuda=True, concat_directions=True):
    """ Encode source sequence. """

    is_lstm = isinstance(encoder, nn.LSTM)

    if h0 is None or c0 is None:
        h0_, c0_ = zero_state(enc_inputs, encoder)
    h0 = h0 if h0 is not None else h0_
    c0 = c0 if c0 is not None else c0_

    mask = (lens > 0)
    if not mask.any():
        h_in = enc_inputs.new_zeros((enc_inputs.size(0), enc_inputs.size(1), encoder.hidden_size * 2 if encoder.bidirectional else encoder.hidden_size))
        hn = h0
        cn = c0
    else:
        batch_size = enc_inputs.size(0)
        enc_inputs = enc_inputs.masked_select(mask.unsqueeze(1).unsqueeze(2)).view(-1, *enc_inputs.size()[1:])
        lens = lens.masked_select(mask)
        h0_ = h0.masked_select(mask.unsqueeze(0).unsqueeze(2)).view(h0.size(0), -1, h0.size(2))
        c0_ = c0.masked_select(mask.unsqueeze(0).unsqueeze(2)).view(c0.size(0), -1, c0.size(2))

        packed_inputs = nn.utils.rnn.pack_padded_sequence(enc_inputs, lens, batch_first=True, enforce_sorted=False)
        state = (h0_, c0_) if is_lstm else h0_
        packed_h_in, state = encoder(packed_inputs, state)

        if is_lstm:
            hn, cn = state
        else:
            hn = state

        h_in, _ = nn.utils.rnn.pad_packed_sequence(packed_h_in, batch_first=True)

        hn = (h0 * 0).masked_scatter(mask.unsqueeze(0).unsqueeze(2), hn)

        if is_lstm:
            cn = (c0 * 0).masked_scatter_(mask.unsqueeze(0).unsqueeze(2), cn)

        h_in = h_in.new_zeros((batch_size, *h_in.size()[1:])).masked_scatter(mask.unsqueeze(1).unsqueeze(2), h_in)

    hn = hn[-2:]
    cn = cn[-2:] if is_lstm else None
    if encoder.bidirectional and concat_directions:
        hn = hn.view(hn.size(0) // 2, 2, hn.size(1), hn.size(2)).transpose(1, 2).contiguous().view(hn.size(0) // 2, hn.size(1), hn.size(2) * 2)
        cn = cn.view(cn.size(0) // 2, 2, cn.size(1), cn.size(2)).transpose(1, 2).contiguous().view(cn.size(0) // 2, cn.size(1), cn.size(2) * 2) if is_lstm else None

    state = (hn, cn) if is_lstm else hn

    return h_in, state

class SimpleSelfAttention(nn.Module):
    def __init__(self, hidden_dim, separate_first=False):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.V = nn.Linear(self.hidden_dim * 2, 1)
        self.W_out = nn.Linear(self.hidden_dim * 3 if separate_first else self.hidden_dim * 2, self.hidden_dim)
        self.separate_first = separate_first

    def forward(self, h, mask, hn):

        h_to_attn = self.W1(h)
        hn_to_attn = self.W2(hn)

        attn_w = self.V(torch.tanh(h_to_attn + hn_to_attn.unsqueeze(1)))

        if self.separate_first:
            m1 = attn_w.new_zeros((attn_w.size(1), 1), dtype=torch.bool)
            m1[0] = 1
            attn_w = attn_w.masked_fill(m1, -1e12)
            attn = torch.softmax(attn_w.masked_fill(mask.unsqueeze(-1), -1e20), 1)
            attn = attn.masked_fill(m1, 0)
            return self.W_out(torch.cat([attn.transpose(1, 2).bmm(h), h[:, :1], hn.unsqueeze(1)], -1))
        else:
            attn = torch.softmax(attn_w.masked_fill(mask.unsqueeze(-1), -1e12), 1)

            return self.W_out(torch.cat([attn.transpose(1, 2).bmm(h), hn.unsqueeze(1)], -1))

class Seq2SeqModel(nn.Module):
    """
    A complete encoder-decoder model, with optional attention.
    """
    def __init__(self, args, emb_matrix=None, use_cuda=False):
        super().__init__()
        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.hidden_dim = args['hidden_dim']
        self.nlayers = args['num_layers'] # encoder layers, decoder layers = 1
        self.emb_dropout = args.get('emb_dropout', 0.0)
        self.dropout = args['dropout']
        self.pad_token = constant.PAD_ID
        self.max_dec_len = args['max_dec_len']
        self.use_cuda = use_cuda
        self.top = args.get('top', 1e10)
        self.args = args
        self.emb_matrix = emb_matrix

        self.gumbel = torch.distributions.gumbel.Gumbel(0, 1)

        print("Building an attentional Seq2Seq model...")
        self.num_directions = 2
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim

        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.drop = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)

        self.encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, self.nlayers, \
            bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.encoder2 = nn.LSTM(self.hidden_dim, self.enc_hidden_dim, self.nlayers, \
            bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.bg_selfattn = SimpleSelfAttention(self.enc_hidden_dim * 2)

        self.h_bg_to_h = nn.Linear(self.enc_hidden_dim * 2, self.enc_hidden_dim * 2)
        self.c_bg_to_c = nn.Linear(self.enc_hidden_dim * 2, self.enc_hidden_dim * 2)

        self.pair_encoder = nn.LSTM(self.enc_hidden_dim * 2, self.enc_hidden_dim * 2, 1,
            batch_first=True)

        dec_input_dim = self.emb_dim

        enc_output_dim = self.enc_hidden_dim * 6

        self.h_enc2dec = nn.Linear(enc_output_dim, self.hidden_dim)
        self.c_enc2dec = nn.Linear(enc_output_dim, self.hidden_dim)

        self.pair_level_attn = SimpleSelfAttention(self.enc_hidden_dim * 2, separate_first=True)

        dec_input_dim += self.hidden_dim # representation of background
        h_bg_dim = self.enc_hidden_dim * 2
        h_bg_dim += self.enc_hidden_dim * 2 # representation of conversational history
        self.h_bg_linear = nn.Linear(h_bg_dim, self.hidden_dim)

        self.decoder = LSTMAttention(dec_input_dim, self.dec_hidden_dim, 1, \
                dropout=args['dropout'],
                batch_first=True, attn_type=self.args['attn_type'], hidden_size2=self.hidden_dim)

        self.emb2dec_out = nn.Linear(self.emb_dim, self.dec_hidden_dim)
        self.dec2vocab_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.dec2vocab = lambda x: self.dec2vocab_bias + x.matmul(torch.tanh(self.emb2dec_out(self.embedding.weight)).transpose(0, 1))

        self.init_weights()

    def init_weights(self):
        # initialize embeddings
        init_range = constant.EMB_INIT_RANGE
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == (self.vocab_size, self.emb_dim), \
                    "Input embedding matrix must match size: {} x {}".format(self.vocab_size, self.emb_dim)
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            self.embedding.weight.data.uniform_(-init_range, init_range)
        # decide finetuning
        if self.top <= 0:
            print("Do not finetune embedding layer.")
            self.embedding.weight.requires_grad = False
        elif self.top < self.vocab_size:
            print("Finetune top {} embeddings.".format(self.top))
            self.embedding.weight.register_hook(lambda x: keep_partial_grad(x, self.top))
        else:
            print("Finetune all embeddings.")
            self.embedding.weight.register_hook(lambda x: keep_partial_grad(x))

    def cuda(self):
        super().cuda()
        self.use_cuda = True

    def cpu(self):
        super().cpu()
        self.use_cuda = False


    def decode(self, dec_inputs, hn, cn, ctx, ctx_mask=None, pair_level_output=None, turn_ids=None, previous_output=None, decoder=None, h_bg=None):
        """ Decode a step, based on context encoding and source context states."""
        decoder = self.decoder if decoder is None else decoder
        dec_hidden = (hn, cn)
        h_out, dec_hidden, attn = decoder(dec_inputs, dec_hidden, ctx, ctx_mask, turn_ids=turn_ids, pair_level_output=pair_level_output, previous_output=previous_output, h_bg=h_bg)

        h_out_reshape = h_out.contiguous().view(h_out.size(0) * h_out.size(1), -1)
        decoder_logits = self.dec2vocab(self.drop(h_out_reshape))
        decoder_logits = decoder_logits.view(h_out.size(0), h_out.size(1), -1)
        log_probs = self.get_log_prob(decoder_logits)
        return log_probs, dec_hidden, attn, h_out

    def encode_sources(self, src, src_mask, bg, bg_mask, encoder=None, bg_encoder=None, h_enc2dec=None, c_enc2dec=None, embedding=None, turn_ids=None):
        encoder = self.encoder if encoder is None else encoder
        bg_encoder = self.bg_encoder if bg_encoder is None and hasattr(self, 'bg_encoder') else bg_encoder
        embedding = self.embedding if embedding is None else embedding
        h_enc2dec = self.h_enc2dec if h_enc2dec is None else h_enc2dec
        c_enc2dec = self.c_enc2dec if c_enc2dec is None else c_enc2dec
        B, T, L = src.size()
        src = src.view(B*T, L)
        src_lens = src.ne(0).sum(1)
        src_mask = src_mask.view(B*T, -1)
        enc_inputs = self.emb_drop(embedding(src))

        hn_attn_bg = 0
        attn = None
        attn_orig = None
        bg_inputs = self.emb_drop(embedding(bg))
        bg_lens = (bg_mask.data.eq(0).long().sum(1))
        h_in2, (hn_bg, cn_bg) = encode(bg_inputs, bg_lens, encoder=encoder, concat_directions=True)
        # attention over background words
        hn_attn_bg = self.bg_selfattn(self.drop(h_in2), bg_mask, self.drop(hn_bg.squeeze(0)))

        hn2 = (self.h_bg_to_h(self.drop(hn_bg)))
        cn2 = self.c_bg_to_c(self.drop(cn_bg))

        # broadcast
        hn2 = hn2.unsqueeze(2).expand(1, B, T, -1).contiguous().view(1, B*T, -1)
        cn2 = cn2.unsqueeze(2).expand(1, B, T, -1).contiguous().view(1, B*T, -1)
        hn_attn_bg = hn_attn_bg.expand(B, T, -1).contiguous().view(B*T, 1, -1)

        hn2 = torch.cat(hn2.split(hn2.size(-1) // 2, dim=-1), 0)
        cn2 = torch.cat(cn2.split(cn2.size(-1) // 2, dim=-1), 0)

        h_in, (hn, cn) = encode(enc_inputs, src_lens, encoder=encoder, h0=hn2, c0=cn2)

        pair_lens = torch.arange(1, T+1, device=src.device).view(1, -1).expand(B, T).contiguous().view(-1)
        pair_input = hn.view(B, T, hn.size(-1))
        pair_lens = pair_lens.masked_fill(src_lens == 0, 0)
        pair_maxlens = pair_lens.view(B, T).max(1)[0]

        h_pairs, (hn, cn) = encode(self.drop(pair_input), pair_maxlens, encoder=self.pair_encoder)

        pair_mask = torch.arange(0, T, device=src.device).view(1, -1) >= pair_lens.view(-1, 1)
        h_in2 = h_in2.unsqueeze(1).expand(B, T, h_in2.size(1), h_in2.size(2)).contiguous().view(B*T, h_in2.size(1), h_in2.size(2))
        bg_mask = bg_mask.unsqueeze(1).expand(B, T, bg_mask.size(1)).contiguous().view(B*T, bg_mask.size(1))

        h_in = h_in.view(B, T, L, -1) + h_pairs.view(B, T, 1, -1)
        h_in = h_in.view(B, 1, T, L, -1).expand(B, T, T, L, -1).contiguous().view(B*T, T*L, -1)
        src_mask = src_mask.view(B, 1, T, L).expand(B, T, T, L).contiguous()
        src_mask = src_mask.masked_fill(pair_mask.view(B, T, T).unsqueeze(-1), 1).view(B*T, T*L) # make sure we don't leak the future

        h_in = torch.cat([h_in2, h_in], 1)
        src_mask = torch.cat([bg_mask, src_mask], 1)

        h_bg = hn_attn_bg.squeeze(1)
        h_pairs_expanded = self.drop(h_pairs).view(B, 1, T, -1).expand(B, T, T, h_pairs.size(-1)).contiguous().view(B*T, T, -1)
        attn_out = self.pair_level_attn(h_pairs_expanded, pair_mask, self.drop(h_pairs).view(B*T, -1))

        h_bg = torch.cat([h_bg, attn_out.squeeze(1)], -1)

        hn = h_enc2dec(self.drop(torch.cat([h_pairs.view(1, B*T, -1), h_bg.unsqueeze(0)], -1)))
        cn = c_enc2dec(self.drop(torch.cat([h_pairs.view(1, B*T, -1), h_bg.unsqueeze(0)], -1)))

        h_bg = self.h_bg_linear(self.drop(h_bg))

        return h_in, src_mask, (hn, cn), h_bg

    def forward(self, src, src_mask, turn_ids, tgt_in, bg=None, bg_mask=None, neg_in=None, neg_out=None, tgt_out=None, reward_only=False, ctx=None):
        # prepare for encoder/decoder
        B, T, L = tgt_in.size()
        tgt_in = tgt_in.view(B*T, L)
        dec_inputs = self.emb_drop(self.embedding(tgt_in))

        h_in, src_mask, (hn, cn), h_bg = self.encode_sources(src, src_mask, bg, bg_mask, turn_ids=turn_ids)

        log_probs, _, dec_attn, _ = self.decode(dec_inputs, hn, cn, h_in, src_mask, turn_ids=turn_ids, h_bg=self.drop(h_bg))

        return log_probs

    def get_log_prob(self, logits):
        logits_reshape = logits.view(-1, self.vocab_size)
        log_probs = torch.log_softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return log_probs
        return log_probs.view(logits.size(0), logits.size(1), logits.size(2))

    def predict(self, src, src_mask, turn_ids, beam_size=5, bg=None, bg_mask=None, return_pair_level=False):
        """ Predict with beam search. """
        batch_size = src.size(0) * src.size(1)

        # (1) encode source
        h_in, src_mask, (hn, cn), h_bg = self.encode_sources(src, src_mask, bg, bg_mask, turn_ids=turn_ids)

        # (2) set up beam
        with torch.no_grad():
            h_in = h_in.data.repeat(beam_size, 1, 1) # repeat data for beam search
            src_mask = src_mask.repeat(beam_size, 1)
            h_bg = h_bg.repeat(beam_size, 1)
            # repeat decoder hidden states
            hn = hn.data.repeat(beam_size, 1, 1)
            cn = cn.data.repeat(beam_size, 1, 1)
        beam = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]

        def update_state(states, idx, positions, beam_size):
            """ Select the states according to back pointers. """
            for e in states:
                bl, batch_size, d = e.size()
                s = e.contiguous().view(beam_size, bl // beam_size, batch_size, d)[:,:,idx]
                s.data.copy_(s.data.index_select(0, positions))

        # (3) main loop
        hids = []
        h_bg = self.drop(h_bg)
        for i in range(self.max_dec_len):
            dec_inputs = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            dec_inputs = self.emb_drop(self.embedding(dec_inputs))

            log_probs, (hn, cn), _, h_out = self.decode(dec_inputs, hn, cn, h_in, src_mask,
                turn_ids=turn_ids, previous_output=None if len(hids) == 0 else hids, h_bg=h_bg)
            hids.append(h_out.squeeze(1))
            log_probs = log_probs.view(beam_size, batch_size, -1).transpose(0,1)\
                    .contiguous() # [batch, beam, V]

            # advance each beam
            done = []
            for b in range(batch_size):
                is_done = beam[b].advance(log_probs.data[b])
                if is_done:
                    done += [b]
                # update beam state
                update_state((hn, cn), b, beam[b].get_current_origin(), beam_size)

            if len(done) == batch_size:
                break

        # back trace and find hypothesis
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[0]]
            k = ks[0]
            hyp = beam[b].get_hyp(k)
            hyp = seq2sequtils.prune_hyp(hyp)
            all_hyp += [hyp]

        return all_hyp

    def sample(self, src, src_mask, turn_ids, top_p=1, bg=None, bg_mask=None, return_pair_level=False, return_logprobs=False):
        """ Top-p sampling """
        batch_size = src.size(0) * src.size(1)

        # (1) encode source
        h_in, src_mask, (hn, cn), h_bg = self.encode_sources(src, src_mask, bg, bg_mask, turn_ids=turn_ids)

        # (2) initialize start of sequence
        dec_inputs = cn.new_full((batch_size, 1), SOS_ID, dtype=torch.long)

        # (3) main loop
        preds = []
        hids = []
        ended = src.new_zeros((batch_size, ), dtype=torch.bool)
        nll = src.new_zeros((batch_size, ), dtype=torch.float32)
        h_bg = self.drop(h_bg)
        for i in range(self.max_dec_len):
            dec_inputs = self.emb_drop(self.embedding(dec_inputs))

            log_probs, (hn, cn), _, h_out = self.decode(dec_inputs, hn, cn, h_in, src_mask,
                turn_ids=turn_ids, previous_output=None if len(hids) == 0 else hids, h_bg=h_bg)
            hids.append(h_out.squeeze(1))
            log_probs = log_probs.view(batch_size, -1)

            if top_p < 1:
                sorted_idx = torch.argsort(-log_probs, dim=1)
                cumprob = torch.cumsum(torch.exp(log_probs.gather(1, sorted_idx)), 1)
                mask = (cumprob > torch.max(torch.exp(torch.max(log_probs, 1, keepdim=True)[0]), log_probs.new_full((batch_size, 1), top_p)))
                mask.masked_fill_(torch.exp(torch.max(log_probs, 1, keepdim=True)[0]) == 1, 1)
                mask[:, 0] = 0
                inverse_idx = torch.argsort(sorted_idx, dim=1)
                inverted_mask = mask.gather(1, inverse_idx)
                log_probs.masked_fill_(inverted_mask, -1e12)

            probs = torch.softmax(log_probs, 1) # renormalize top p

            m = Categorical(probs)

            dec_inputs = m.sample()
            nll = torch.where(ended, nll, nll - m.log_prob(dec_inputs))

            ended = ended | (dec_inputs == EOS_ID)

            dec_inputs = dec_inputs.unsqueeze(1)
            preds.append(dec_inputs)

        preds = torch.cat(preds, 1).tolist()

        return preds, nll

class CompositeEmbedding(nn.Module):
    def __init__(self, args, dropout=0, emb_matrix=None, use_cuda=False, use_elmo=False):
        super().__init__()
        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.pad_token = constant.PAD_ID
        self.use_cuda = use_cuda
        self.top = args.get('top', 1e10)
        self.args = args
        self.emb_matrix = emb_matrix
        self.use_elmo = use_elmo

        self.drop = nn.Dropout(dropout)

        self.char_embedding = nn.Embedding(args['char_vocab_size'], args['char_emb_dim'], padding_idx=self.pad_token)
        self.char_embedding.weight.register_hook(lambda x: keep_partial_grad(x))
        self.char_conv = nn.Conv1d(args['char_emb_dim'], args['char_hidden_dim'], args['char_conv_size'], padding=args['char_conv_size'])

        if use_elmo:
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

            self.elmo = Elmo(options_file, weight_file, 1, dropout=dropout, vocab_to_cache=args['vocab']['id2word'])
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.pad_token)
            self.init_weights()

    def init_weights(self):
        # initialize embeddings
        init_range = constant.EMB_INIT_RANGE
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == self.embedding.weight.size(), \
                    "Input embedding matrix must match size: {} x {}".format(*self.embedding.weight.size())
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            self.embedding.weight.data.uniform_(-init_range, init_range)
        # decide finetuning
        if self.top <= 0:
            print("Do not finetune embedding layer.")
            self.embedding.weight.requires_grad = False
        elif self.top < self.vocab_size:
            print("Finetune top {} embeddings.".format(self.top))
            self.embedding.weight.register_hook(lambda x: keep_partial_grad(x, self.top))
        else:
            print("Finetune all embeddings.")
            self.embedding.weight.register_hook(lambda x: keep_partial_grad(x))

    def forward(self, idx, char_idx, text):
        B, T, L = char_idx.size()
        char_idx = char_idx.view(B*T, L)
        if char_idx.size(0) == 0:
            if self.use_elmo:
                word_dim = 1024
            else:
                word_dim = self.emb_dim
            return char_idx.new_zeros(B, T, word_dim + self.args['char_hidden_dim'], dtype=torch.float32)
        char_inputs = self.drop(self.char_embedding(char_idx))
        char_hid = torch.relu(self.char_conv(char_inputs.transpose(1, 2)))
        char_pooled = char_hid.max(2)[0]
        char_pooled = char_pooled.view(B, T, char_pooled.size(-1))

        if self.use_elmo:
            ids = batch_to_ids(text).contiguous()
            ids = ids.to(idx.device)
            embs = self.elmo(ids, word_inputs=idx)['elmo_representations'][0]
        else:
            embs = (self.embedding(idx))

        return torch.cat([embs, char_pooled], -1)

def pool(hid, mask):
    return hid.masked_fill(mask.unsqueeze(2), -1e12).max(1)[0]

class BiAttention(nn.Module):
    def __init__(self, hidden_dim, self_attn=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_attn = self_attn

        self.h1_to_attn = nn.Linear(self.hidden_dim, 1, bias=False)
        self.h2_to_attn = nn.Linear(self.hidden_dim, 1, bias=False)
        std = np.sqrt(6 / (self.hidden_dim + 1))
        self.h12_to_attn = nn.Parameter(torch.Tensor(self.hidden_dim))
        self.h12_to_attn.data.uniform_(-std, std)
        self.bias = nn.Parameter(torch.Tensor([0]))
        output_count = 3 if self_attn else 4
        self.merge_1 = nn.Linear(self.hidden_dim * output_count, self.hidden_dim)

        if not self_attn:
            self.merge_2 = nn.Linear(self.hidden_dim * output_count, self.hidden_dim)

    def forward(self, h1, mask1, h2, mask2):
        attn_w = (self.h12_to_attn * h1).bmm(h2.transpose(1, 2)) + self.h1_to_attn(h1) + self.h2_to_attn(h2).transpose(1, 2) + self.bias
        attn_w.masked_fill_(mask1.unsqueeze(2), -1e12)
        attn_w.masked_fill_(mask2.unsqueeze(1), -1e12)

        if self.self_attn:
            attn_w.masked_fill_(torch.eye(attn_w.size(-1), device=attn_w.device, dtype=torch.bool), -1e12)

        attn_2_to_1 = torch.softmax(attn_w, 2).bmm(h2)

        if self.self_attn:
            merged_1 = torch.relu(self.merge_1((torch.cat([h1, attn_2_to_1, h1 * attn_2_to_1], -1))))
            return merged_1
        else:
            attn_1_to_2 = torch.softmax(attn_w, 1).transpose(1, 2).bmm(h1)

            h1_ = torch.softmax(attn_w.max(2)[0].unsqueeze(1), 2).bmm(h1)
            h2_ = torch.softmax(attn_w.max(1)[0].unsqueeze(1), 2).bmm(h2)

            merged_1 = torch.relu(self.merge_1((torch.cat([h1, attn_2_to_1, h1 * attn_2_to_1, h1 * h1_], -1))))
            merged_2 = torch.relu(self.merge_2((torch.cat([h2, attn_1_to_2, h2 * attn_1_to_2, h2 * h2_], -1))))
            return merged_1, merged_2

def get_logits(self, h_src, h_ctx, h_bg, src_mask, ctx_mask, bg_mask, tgt_out, tgt_out_char, tgt_text, this_turn, ctx_lens, skip_qa_model=False):
    # target lm
    B, T, L, C = tgt_out_char.size()
    tgt_out = tgt_out.view(B*T, L)
    tgt_out_char = tgt_out_char.view(B*T, L, C)
    lens = tgt_out.ne(0).sum(1)
    tgt_in = torch.cat([tgt_out.new_full((tgt_out.size(0), 1), SOS_ID), tgt_out[:, :-1]], 1)
    sos_char = torch.LongTensor([CHAR_START_ID, SOS_ID, CHAR_END_ID] + [0] * (tgt_out_char.size(2) - 3)).to(tgt_out_char.device).unsqueeze(0).unsqueeze(1)
    tgt_in_char = torch.cat([sos_char.repeat(tgt_out_char.size(0), 1, 1), tgt_out_char[:, :-1]], 1)
    inputs = self.embedding(tgt_in, tgt_in_char, [])
    h_lm, _ = encode(self.emb_drop(inputs), lens, self.tgt_lm)
    lm_logits = self.lm_hid_to_vocab(self.drop(h_lm))
    nll = self.lm_crit(lm_logits.view(-1, lm_logits.size(-1)), tgt_out.view(-1))

    tgt_out_char = tgt_out_char.masked_fill(tgt_out.eq(EOS_ID).unsqueeze(-1), PAD_ID)
    tgt_out = tgt_out.masked_fill(tgt_out.eq(EOS_ID), PAD_ID)
    lens = tgt_out.ne(0).sum(1)
    maxlen = max(len(y) for x in tgt_text for y in x)
    tgt_out = tgt_out[:, :maxlen].contiguous()
    tgt_out_char = tgt_out_char[:, :maxlen].contiguous()
    tgt_text_padded = []
    for x in tgt_text:
        tgt_text_padded.extend([[z.lower() for z in y] for y in x])
        tgt_text_padded.extend([[]] * (T - len(x)))

    tgt_inputs = torch.cat([self.embedding(tgt_out, tgt_out_char, tgt_text_padded), self.question_num_embedding(this_turn).unsqueeze(1).expand(tgt_out.size(0), tgt_out.size(1), self.args['turn_emb_dim'] * self.args['teacher_max_history'])], -1)
    h_tgt, _ = encode(self.emb_drop(tgt_inputs), lens, self.ctx_encoder)
    tgt_mask = (tgt_out == 0)[:, :tgt_inputs.size(1)]

    h_ctx0, h_tgt0, h_src0 = h_ctx, h_tgt, h_src

    h_ctx, h_tgt1 = self.biattn1(self.drop(h_ctx0), ctx_mask, self.drop(h_tgt0), tgt_mask)

    if skip_qa_model:
        start_logit = end_logit = yesno_logit = followup_logit = None
    else:
        h_ctx1, _ = encode(self.drop(h_ctx), ctx_lens, self.ctx_encoder2)
        h_ctx1 = self.drop(h_ctx1)
        h_ctx1 = self.selfattn(h_ctx1, ctx_mask, h_ctx1, ctx_mask)

        start_in = self.drop(h_ctx + h_ctx1)
        start_hid, _ = encode(start_in, ctx_lens, self.rnn_start)
        start_logit = self.hid_to_start((start_hid)).squeeze(-1).masked_fill(ctx_mask, -1e12)

        end_in = torch.cat([self.drop(start_hid), start_in], -1)
        end_hid, _ = encode(end_in, ctx_lens, self.rnn_end)
        end_logit = self.hid_to_end((end_hid)).squeeze(-1).masked_fill(ctx_mask, -1e12)
        yesno_logit = self.hid_to_yesno((end_hid))
        followup_logit = self.hid_to_followup((end_hid))

    if h_bg is not None:
        h_src0 = torch.cat([h_bg, h_src0], 1)
        src_mask = torch.cat([bg_mask, src_mask], 1)

    h_tgt2, h_src = self.biattn2(self.drop(h_tgt0), tgt_mask, self.drop(h_src0), src_mask)
    h_tgt_clf = self.htgt_to_hid(torch.cat([h_tgt1, h_tgt2, h_tgt1 * h_tgt2], -1))
    h_tgt_clf = pool(h_tgt_clf, tgt_mask)
    logit = self.hid_to_logit(h_tgt_clf).squeeze(-1)

    #logit = self.hid_to_logit((pool(h_tgt_clf, tgt_mask))).squeeze(-1)

    return logit, start_logit, end_logit, yesno_logit, followup_logit, nll

class TeacherModel(nn.Module):
    def __init__(self, args, emb_matrix=None, use_cuda=False):
        super().__init__()
        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.hidden_dim = args['teacher_hidden_dim']
        self.nlayers = args['num_layers'] # encoder layers, decoder layers = 1
        self.emb_dropout = args.get('teacher_emb_dropout', 0.0)
        self.dropout = args['teacher_dropout']
        self.pad_token = constant.PAD_ID
        self.max_dec_len = args['max_dec_len']
        self.use_cuda = use_cuda
        self.top = args.get('top', 1e10)
        self.args = args
        self.emb_matrix = emb_matrix

        print("Building the QA/classifier teacher model...")
        self.num_directions = 2
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim

        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.drop = nn.Dropout(self.dropout)
        self.embedding = CompositeEmbedding(args, dropout=self.emb_dropout, emb_matrix=emb_matrix, use_cuda=use_cuda, use_elmo=args['teacher_elmo'])

        if args['teacher_elmo']:
            self.embedding_bgsrc = CompositeEmbedding(args, dropout=self.emb_dropout, emb_matrix=emb_matrix, use_cuda=use_cuda)
        else:
            self.embedding_bgsrc = self.embedding

        self.encoder = nn.GRU(self.emb_dim + args['char_hidden_dim'], self.hidden_dim//2, 1, \
            bidirectional=True, batch_first=True)
        self.pair_encoder = nn.GRU(self.hidden_dim, self.hidden_dim, 1,
            batch_first=True)

        self.prev_ans_embedding = nn.Embedding(args['teacher_max_history'] * 4 + 1, args['turn_emb_dim'], 0)
        self.prev_ans_embedding.weight.register_hook(lambda x: keep_partial_grad(x))
        self.question_num_embedding = nn.Embedding(MAX_TURNS, args['turn_emb_dim'] * args['teacher_max_history'])
        self.src_question_num_embedding = nn.Embedding(MAX_TURNS, args['turn_emb_dim'] * args['teacher_max_history'])

        if args.get('teacher_elmo', False):
            self.emb_dim = 1024

        self.emb_dim += args['char_hidden_dim']

        self.ctx_encoder = nn.GRU(self.emb_dim + args['turn_emb_dim'] * args['teacher_max_history'], self.hidden_dim//2, 1, \
            bidirectional=True, batch_first=True)
        self.ctx_encoder2 = nn.GRU(self.hidden_dim, self.hidden_dim//2, 1, \
            bidirectional=True, batch_first=True)

        self.biattn1 = BiAttention(args['teacher_hidden_dim'])
        self.biattn2 = BiAttention(args['teacher_hidden_dim'])
        self.selfattn = BiAttention(args['teacher_hidden_dim'], self_attn=True)
        self.ctx_merge = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.rnn_start = nn.GRU(self.hidden_dim, self.hidden_dim//2, self.nlayers, \
            bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.rnn_end = nn.GRU(self.hidden_dim * 2, self.hidden_dim//2, self.nlayers, \
            bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.rnn_clf = nn.GRU(self.hidden_dim * 2, self.hidden_dim//2, self.nlayers, \
            bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.hid_to_start = nn.Linear(self.hidden_dim, 1)
        self.hid_to_end = nn.Linear(self.hidden_dim, 1)
        self.hid_to_yesno = nn.Linear(self.hidden_dim, len(YESNO_TO_ID))
        self.hid_to_followup = nn.Linear(self.hidden_dim, len(FOLLOWUP_TO_ID))
        self.crit = nn.CrossEntropyLoss(ignore_index=-1)
        self.clf_crit = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.FloatTensor([.5, 1]))

        self.htgt_to_hid = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 3)
        self.hid_to_logit = nn.Linear(self.hidden_dim * 3, 1)

        self.h_bg2enc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c_bg2enc = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.tgt_lm = nn.GRU(self.emb_dim, self.hidden_dim, 1, batch_first=True)
        self.lm_hid_to_vocab = nn.Linear(self.hidden_dim, self.vocab_size)
        self.lm_crit = nn.CrossEntropyLoss(ignore_index=0)

    def cuda(self):
        super().cuda()
        self.use_cuda = True

    def cpu(self):
        super().cpu()
        self.use_cuda = False

    def forward(self, src, src_mask, turn_ids, tgt_in, bg=None, bg_mask=None, neg_in=None, neg_out=None, tgt_out=None, reward_only=False, ctx=None, ans_mask=None, start=None, end=None, this_turn=None,
            src_char=None, tgt_out_char=None, bg_char=None, ctx_char=None, neg_out_char=None, ctx_elmo=None, tgt_out_elmo=None, neg_out_elmo=None,
            yesno=None, followup=None, ctx_text=None, tgt_text=None, src_text=None, neg_text=None, bg_text=None, return_all_rewards=False):

        loss = acc = 0
        reward = None

        h0 = c0 = h_bg = None
        if bg is not None:
            bg_lens = (bg_mask == 0).sum(1)
            bg_emb = self.emb_drop(self.embedding_bgsrc(bg, bg_char, bg_text))
            h_bg, h0 = encode(bg_emb, bg_lens, encoder=self.encoder)

            h0 = self.h_bg2enc(h0)
            h0 = torch.cat(h0.split(h0.size(-1) // 2, -1), 0)

        B, T, L = src.size()
        src = src.view(B*T, L)
        src_char = src_char.view(B*T, L, -1)
        src_mask = src_mask.view(B*T, L)
        src_text_padded = []
        for x in src_text:
            src_text_padded.extend([[z for z in y] for y in x] + [[]] * (T-len(x)))
        src_lens = (src != 0).sum(1)
        src_emb = self.embedding(src, src_char, src_text_padded)
        src_emb = torch.cat([src_emb, self.src_question_num_embedding(this_turn.view(-1)).masked_fill(src_lens.unsqueeze(1) == 0, 0).unsqueeze(1).expand(src.size(0), src.size(1), -1)], -1)
        src_emb = self.emb_drop(src_emb)
        h0 = h0.unsqueeze(2).expand(h0.size(0), B, T, h0.size(-1)).contiguous().view(h0.size(0), -1, h0.size(-1))
        h_in, hn = encode(src_emb, src_lens, encoder=self.ctx_encoder, h0=h0)

        pair_lens = torch.arange(1, T+1, device=src.device).view(1, -1).expand(B, T).contiguous().view(-1)
        pair_input = hn.view(B, T, hn.size(-1))
        pair_lens = pair_lens.masked_fill(src_lens == 0, 0)
        max_pairlens = pair_lens.view(B, T).max(1)[0]

        h_pairs, hn = encode(self.drop(pair_input), max_pairlens, encoder=self.pair_encoder)
        pair_mask = torch.arange(0, T, device=src.device).view(1, -1) >= pair_lens.view(-1, 1)
        h_pairs_expanded = h_pairs.view(B, T, 1, -1)
        h_in = h_in.view(B, T, L, -1)
        h_in = h_in + h_pairs_expanded
        h_in = h_in.view(B, 1, T, L, -1).expand(B, T, T, L, h_in.size(-1)).contiguous().view(B*T, T*L, -1)
        src_mask = src_mask.view(B, 1, T, L).expand(B, T, T, L).contiguous()
        src_mask = src_mask.masked_fill(pair_mask.view(B, T, T).unsqueeze(-1), 1).view(B*T, T*L) # make sure we don't leak the future

        ctx_emb = (self.embedding(ctx, ctx_char, [[w for w in p] for p in ctx_text]))

        ctx = ctx.unsqueeze(1).expand(B, T, -1).contiguous().view(B*T, -1)
        ctx_mask = (ctx == 0)
        ans_mask = ans_mask.view(B*T, ans_mask.size(2), ans_mask.size(3))
        this_turn = this_turn.view(-1)

        rng = torch.arange(MAX_TURNS + self.args['teacher_max_history'], device=this_turn.device).unsqueeze(0)
        m = (rng < this_turn.view(-1, 1) + self.args['teacher_max_history']) & (rng >= this_turn.view(-1, 1))
        ans_mask1 = torch.cat([ans_mask.new_zeros((ans_mask.size(0), ans_mask.size(1), self.args['teacher_max_history'])), ans_mask], -1)
        ans_idx = (ans_mask1 + ((this_turn.view(-1, 1) + self.args['teacher_max_history'] - rng - 1) * 4).unsqueeze(1)).long()
        ans_idx = ans_idx.masked_fill((ctx == 0).unsqueeze(2), 0).masked_fill(rng.unsqueeze(0) < self.args['teacher_max_history'], 0).masked_fill(ans_mask1 == 0, 0)
        ans_idx = ans_idx.masked_select(m.unsqueeze(1)).long().view(ans_mask.size(0), ans_mask.size(1), self.args['teacher_max_history'])

        ctx_emb = ctx_emb.unsqueeze(1).expand(B, T, ctx_emb.size(1), ctx_emb.size(2)).contiguous().view(B*T, ctx_emb.size(1), ctx_emb.size(2))

        ctx_emb = torch.cat([ctx_emb, (self.prev_ans_embedding(ans_idx)).view(ans_mask.size(0), ans_mask.size(1), -1)], -1)
        ctx_lens = (ctx != 0).sum(1)
        h_ctx, _ = encode(self.emb_drop(ctx_emb), ctx_lens, encoder=self.ctx_encoder)


        B, T, L, C = tgt_out_char.size()
        max_turns = this_turn.view(B, T).max(1, keepdim=True)[0]
        idx = torch.randint(32768, (B, T), device=this_turn.device) % max_turns
        idx = torch.where(idx >= this_turn.view(B, T), idx+1, idx)
        neg2_out = tgt_out.gather(1, idx.unsqueeze(2).expand(B, T, tgt_out.size(2)))
        neg2_out_char = tgt_out_char.gather(1, idx.unsqueeze(2).unsqueeze(3).expand(B, T, tgt_out.size(2), tgt_out_char.size(3)))
        maxlen = neg2_out.ne(0).sum(2).max()
        neg2_out = neg2_out[:, :, :maxlen].contiguous()
        neg2_out_char = neg2_out_char[:, :, :maxlen].contiguous()
        neg2_text = [[tgt_text[i][j] for j in x] for i, x in enumerate(idx.tolist())]

        if bg is not None:
            h_bg = h_bg.unsqueeze(1).repeat(1, T, 1, 1).contiguous().view(-1, h_bg.size(1), h_bg.size(2))
            bg_mask = bg_mask.unsqueeze(1).repeat(1, T, 1).contiguous().view(-1, bg_mask.size(1))

        # randomize evaluation order to counter the effect of AllenNLP ELMo's statefulness
        logit_pos, start_logits_pos, end_logits_pos, yesno_pos, followup_pos, nll_pos = get_logits(self, h_in, h_ctx, h_bg, src_mask, ctx_mask, bg_mask, tgt_out, tgt_out_char, tgt_text, this_turn, ctx_lens)
        logit_neg, start_logits_neg, end_logits_neg, _, _, nll_neg = get_logits(self, h_in, h_ctx, h_bg, src_mask, ctx_mask, bg_mask, neg_out, neg_out_char, neg_text, this_turn, ctx_lens, skip_qa_model=not reward_only)
        logit_neg2, start_logits_neg2, end_logits_neg2, _, _, _ = get_logits(self, h_in, h_ctx, h_bg, src_mask, ctx_mask, bg_mask, neg2_out, neg2_out_char, neg2_text, this_turn, ctx_lens, skip_qa_model=not reward_only)

        start = start.view(-1)
        end = end.view(-1)
        yesno = yesno.view(-1)
        followup = followup.view(-1)

        start = start.masked_fill(start >= ctx.size(1), -1)
        end = (end-1).masked_fill(end-1 >= ctx.size(1), -1).masked_fill(end < 0, -1)
        cannotanswer = (start == end) & (start == (ctx_lens - 1))

        def start_end_logits_to_offsets(start_logits, end_logits, max_span_length=30):
            batch_size, paragraph_length = start_logits.size()
            st = torch.zeros((batch_size, ), dtype=torch.long)
            en = torch.zeros((batch_size, ), dtype=torch.long)
            span_start_argmax = [0] * batch_size
            max_span_log_prob = [-1e20] * batch_size

            span_start_logits = start_logits.data.cpu().numpy()
            span_end_logits = end_logits.data.cpu().numpy()
            for b_i in range(batch_size):
                for j in range(paragraph_length):
                    val1 = span_start_logits[b_i, span_start_argmax[b_i]]
                    if val1 < span_start_logits[b_i, j]:
                        span_start_argmax[b_i] = j
                        val1 = span_start_logits[b_i, j]
                    val2 = span_end_logits[b_i, j]
                    if val1 + val2 > max_span_log_prob[b_i]:
                        if j - span_start_argmax[b_i] > max_span_length:
                            continue
                        st[b_i] = span_start_argmax[b_i]
                        en[b_i] = j
                        max_span_log_prob[b_i] = val1 + val2

            st = st.to(start_logits.device)
            en = en.to(end_logits.device)
            return st, en

        def offsets_to_f1(st_gold, en_gold, st_pred, en_pred, ctx_text):
            f1 = []
            max_turns = st_gold.size(0) // len(ctx_text)
            for idx, (sg, eg, sp, ep) in enumerate(zip(st_gold.cpu(), en_gold.cpu(), st_pred.cpu(), en_pred.cpu())):
                ct = ctx_text[idx // max_turns]
                g = ct[sg:eg+1]
                p = ct[sp:ep+1]
                f1.append(cached_f1(' '.join(p), ' '.join(g)))
            f1 = torch.Tensor(f1).to(st_gold.device)

            return f1

        start_pos, end_pos = start_end_logits_to_offsets(start_logits_pos, end_logits_pos)
        cannotanswer_pos = (start_pos == end_pos) & (end_pos == (ctx_lens - 1)) # last token is CANNOTANSWER

        example_mask = (yesno >= 0)

        if not reward_only:
            reward_pos = reward_neg = 0
            logits0 = torch.cat([logit_pos, logit_neg, logit_neg2], 0)
            logits = torch.cat([logits0.new_zeros(logits0.size(0), 1), logits0.unsqueeze(1)], 1)
            labels = torch.cat([logit_pos.new_ones(logit_pos.size(), dtype=torch.long).masked_fill(src_lens == 0, -1),
                logit_neg.new_zeros(logit_neg.size(), dtype=torch.long).masked_fill(src_lens == 0, -1),
                logit_neg2.new_zeros(logit_neg2.size(), dtype=torch.long).masked_fill(src_lens == 0, -1)], 0)

            tp = (logits0 > 0)[:logit_pos.size(0)].masked_select(example_mask).float().sum()
            prec = tp / ((logits0 > 0).float().sum() + 1e-12)
            recall = tp / ((labels > 0).sum().float() + 1e-12)
            clf_f1 = 2 * prec * recall / (prec + recall + 1e-12)

            f1 = offsets_to_f1(start, end, start_pos, end_pos, ctx_text)
            acc = (clf_f1.item(), f1.masked_select(example_mask).mean().item(), -nll_pos.item())

            end1 = end.masked_fill(end < 0, ctx.size(1)-1)

            loss = self.clf_crit(logits, labels) + self.crit(start_logits_pos, start) + self.crit(end_logits_pos, end)

            yesno = yesno.masked_fill(end < 0, -1)
            followup = followup.masked_fill(end < 0, -1)

            yesno_pos = torch.gather(yesno_pos, 1, end1.unsqueeze(1).unsqueeze(2).expand(end.size(0), 1, yesno_pos.size(-1))).squeeze(1)
            followup_pos = torch.gather(followup_pos, 1, end1.unsqueeze(1).unsqueeze(2).expand(end.size(0), 1, followup_pos.size(-1))).squeeze(1)

            loss += self.crit(yesno_pos, yesno) + self.crit(followup_pos, followup) + nll_pos
        else:
            f1 = offsets_to_f1(start, end, start_pos, end_pos, ctx_text)
            acc = (acc, f1.masked_select(example_mask).mean().item())

            non_example_mask = example_mask.bitwise_not()
            logit_pos = torch.sigmoid(logit_pos).masked_fill(non_example_mask, 0)

            # overlap with existing answers
            start_neg, end_neg = start_end_logits_to_offsets(start_logits_neg, end_logits_neg)
            cannotanswer_neg = (start_neg == end_neg) & (end_neg == (ctx_lens - 1)) # last token is CANNOTANSWER
            logit_neg = torch.sigmoid(logit_neg).masked_fill(non_example_mask, 0)

            ans_mask = ans_mask > 0
            t = torch.arange(ans_mask.size(1), device=ans_mask.device).unsqueeze(1).unsqueeze(0).expand(ans_mask.size())
            en = t.masked_fill(ans_mask.bitwise_not(), -1).max(1)[0]
            st = t.masked_fill(ans_mask.bitwise_not(), 1e10).min(1)[0]
            def max_prec(st_gold, en_gold, st_pred, en_pred, cannotanswer, cannotanswer_pred, ctx_text):
                batch_size = ans_mask.size(0)
                max_turns = batch_size // len(ctx_text)

                prec = []

                cannotanswer_mask = (cannotanswer_pred).cpu()

                for idx, (sg, eg, sp, ep, cm) in enumerate(zip(st_gold.cpu(), en_gold.cpu(), st_pred.cpu(), en_pred.cpu(), cannotanswer_mask)):
                    if cm:
                        prec.append(1)
                        continue
                    ct = ctx_text[idx // max_turns]
                    p = ct[sp:ep+1]

                    golds = []
                    for j in range(ans_mask.size(-1)):
                        if eg[j] < 0:
                            break
                        golds.append(' '.join(ct[sg[j]:eg[j]+1]))

                    if len(golds) == 0:
                        prec.append(0)
                    else:
                        prec.append(squad_eval.metric_max_over_ground_truths(cached_prec, ' '.join(p), golds))

                prec = torch.Tensor(prec).to(ans_mask.device)

                return prec
            pos_novelty = 1 - max_prec(st, en, start_pos, end_pos, cannotanswer, cannotanswer_pos, ctx_text)
            neg_novelty = 1 - max_prec(st, en, start_neg, end_neg, cannotanswer, cannotanswer_neg, ctx_text)

            pos_novelty = pos_novelty.masked_fill(non_example_mask, 0)
            neg_novelty = neg_novelty.masked_fill(non_example_mask, 0)

            logit_adv = logit_pos - logit_neg
            novelty_adv = pos_novelty - neg_novelty
            if self.args['teacher_spec_only']:
                reward_pos = logit_pos
                reward_neg = logit_neg
            elif self.args['teacher_info_only']:
                reward_pos = pos_novelty
                reward_neg = neg_novelty
            else:
                reward_pos = self.args['lambda1'] * logit_pos + (1 - self.args['lambda1']) * pos_novelty
                reward_neg = self.args['lambda1'] * logit_neg + (1 - self.args['lambda1']) * neg_novelty

        if return_all_rewards:
            return loss, acc, reward_pos, reward_neg, logit_pos, logit_neg, pos_novelty, neg_novelty, nll_pos, nll_neg
        else:
            return loss, acc, reward_pos, reward_neg
