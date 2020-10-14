"""
This file contains the main trainer class that's in charge of training and prediction, as well as helper functions.
"""

import random
import sys
import numpy as np
from collections import Counter
import torch
from torch import nn
import torch.nn.init as init
from colors import yellow

import stanza.models.common.seq2seq_constant as constant
from models.seq2seq import Seq2SeqModel, TeacherModel
from models.data import pad_char_start_end
from stanza.models.common import utils

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = {k: batch[k].cuda() if batch[k] is not None and isinstance(batch[k], torch.Tensor) else batch[k] for k in batch}
    else:
        inputs = batch
    return inputs

def samples_to_in_out(sample, vocab, device, batch_size):
    maxlen = 0
    maxcharlen = 0
    s_in = []
    s_out = []
    char_out = []
    text = []
    for s in sample:
        s1 = [constant.SOS_ID]
        for w in s:
            s1.append(w)
            if w == constant.EOS_ID or w == constant.PAD_ID:
                break
        if s1[-1] != constant.EOS_ID:
            s1.append(constant.EOS_ID)
        s_in.append(s1[:-1])
        s_out.append(s1[1:])
        char_out.append(pad_char_start_end([vocab['wordid2chars'][w] for w in s1[1:]]))
        text.append([vocab['id2word'][w] for w in s1[1:-1]])

        maxlen = max(maxlen, len(s1) - 1)
        maxcharlen = max(maxcharlen, max(len(c) for c in char_out[-1]))

    s_in = [x + [constant.PAD_ID] * (maxlen - len(x)) for x in s_in]
    s_out = [x + [constant.PAD_ID] * (maxlen - len(x)) for x in s_out]
    char_out = [[chars + [constant.PAD_ID] * (maxcharlen - len(chars)) for chars in x]
            + [[constant.PAD_ID] * maxcharlen] * (maxlen - len(x)) for x in char_out]

    s_in = torch.tensor(s_in).to(device)
    s_out = torch.tensor(s_out).to(device)
    char_out = torch.tensor(char_out).to(device)

    s_in = s_in.view(batch_size, -1, s_in.size(1))
    s_out = s_out.view(batch_size, -1, s_out.size(1))
    char_out = char_out.view(batch_size, -1, char_out.size(1), char_out.size(2))
    text = [text[st:st+len(text) // batch_size] for st in range(0, len(text), len(text) // batch_size)]
    return s_in, s_out, char_out, text

def get_reward(tgt_in, tgt_out, tgt_out_char, tgt_text, neg_in, neg_out, neg_out_char, neg_text, inputs, teacher, return_all_rewards=False):
    res = teacher(inputs['src'], inputs['src'] == 0, inputs['turn_ids'], tgt_in, bg=inputs['bg'], bg_mask=(inputs['bg'] == 0),
            neg_in=neg_in, neg_out=neg_out, tgt_out=tgt_out, ctx=inputs['ctx'], ans_mask=inputs['ans_mask'], start=inputs['start'], end=inputs['end'],
            this_turn=inputs['this_turn'], src_char=inputs['src_char'], tgt_out_char=tgt_out_char, ctx_char=inputs['ctx_char'], bg_char=inputs.get('bg_char', None),
            neg_out_char=neg_out_char, yesno=inputs['yesno'], followup=inputs['followup'], src_text=inputs['src_text'], bg_text=inputs['bg_text'], tgt_text=tgt_text, neg_text=neg_text, ctx_text=inputs['ctx_text'], reward_only=True, return_all_rewards=return_all_rewards)
    return res[2:]

def filter_elmo(state_dict):
    for k in list(state_dict.keys()):
        if 'elmo' in k and 'scalar_mix' not in k:
            del state_dict[k]
    return state_dict

class Seq2SeqTrainer(object):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, use_cuda=False, teacher_model_file=None):
        self.use_cuda = use_cuda
        self.vocab = vocab
        if model_file is not None:
            # load everything from file
            self.load(model_file, use_cuda, args=args, vocab=vocab)
        else:
            # build model from scratch
            self.args = args
            self.model = Seq2SeqModel(args, emb_matrix=emb_matrix, use_cuda=use_cuda)
            # dict-based components
            self.word_dict = dict()
            self.composite_dict = dict()
        self.crit = torch.nn.NLLLoss(ignore_index=constant.PAD_ID)
        self.nce_parameters = []
        if self.args['lambda2'] != 0:
            self.init_teacher(teacher_model_file, use_cuda, emb_matrix=emb_matrix)
        self.teacher_update_count = 0
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if use_cuda:
            self.model.cuda()
            self.crit.cuda()
        else:
            self.model.cpu()
            self.crit.cpu()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'])

    def init_teacher(self, teacher_model_file, use_cuda, emb_matrix):
        if teacher_model_file is not None:
            self.load_teacher(teacher_model_file, use_cuda)
        else:
            self.teacher = TeacherModel(self.args, emb_matrix=emb_matrix, use_cuda=use_cuda)
        self.teacher_parameters = [p for p in self.teacher.parameters() if p.requires_grad]
        self.teacher_optimizer = torch.optim.SGD(self.teacher_parameters, lr=0.01, momentum=0.9)

        if use_cuda:
            self.teacher.cuda()
        else:
            self.teacher.cpu()

    def update(self, batch, eval=False, freeze_teacher=False, i = 0):
        inputs = unpack_batch(batch, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids, ctx = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids'], inputs['ctx']
        bg = inputs.get('bg', None)
        src_mask = src.eq(constant.PAD_ID)
        bg_mask = bg.eq(constant.PAD_ID) if bg is not None else None

        if eval:
            self.model.eval()
            if self.args['lambda2'] != 0:
                self.teacher.eval()
        else:
            self.model.train()
            if i % 2 == 0:
                self.optimizer.zero_grad()
            if self.args['lambda2'] != 0:
                if not freeze_teacher:
                    self.teacher.train()
                else:
                    self.teacher.eval()
                if not freeze_teacher and i % 2 == 0:
                    self.teacher_optimizer.zero_grad()

        if self.args['lambda2'] != 0:
            self.teacher_update_count += 1

            sr_mean = 0
            if self.args['lambda_reinforce'] != 0:
                sample, s_nll = self.sample(batch, return_preds=True, top_p=self.args['top_p'])

                neg_in, neg_out, neg_out_char, neg_text = samples_to_in_out(sample, self.vocab, src.device, src.size(0))
                s_nll /= neg_out.ne(0).sum() / neg_out.size(0)
                with torch.no_grad():
                    greedy = self.predict(batch, return_preds=True)
                    greedy_in, greedy_out, greedy_out_char, greedy_text = samples_to_in_out(greedy, self.vocab, src.device, src.size(0))
                    if max(len(y) for x in neg_text for y in x) == 0 or max(len(y) for x in greedy_text for y in x) == 0:
                        return 0, (0, 0, 0), 0
                    s_r, g_r, s_flow, g_flow, s_novelty, g_novelty, s_extnll, g_extnll = get_reward(neg_in, neg_out, neg_out_char, neg_text, greedy_in, greedy_out, greedy_out_char, greedy_text, inputs, self.teacher, return_all_rewards=True)
                    sr_mean = g_r.masked_select((tgt_out > 0).any(2).view(-1)).mean() - .1 * g_extnll.item()
            else:
                # random frequent question
                neg_in = None
                neg_out = inputs['neg_out']
                neg_out_char = inputs['neg_out_char']
                neg_text = inputs['neg_text']

            if freeze_teacher:
                teacher_loss = 0
                teacher_acc = (g_flow.masked_select((tgt_out > 0).any(2).view(-1)).mean().item(),
                        g_novelty.masked_select((tgt_out > 0).any(2).view(-1)).mean().item(),
                        -g_extnll.item())
            else:
                teacher_loss, teacher_acc = self.teacher(src, src_mask, turn_ids, tgt_in, bg=bg, bg_mask=bg_mask, neg_in=neg_in, neg_out=neg_out, tgt_out=tgt_out, ctx=ctx, ans_mask=inputs['ans_mask'], start=inputs['start'], end=inputs['end'], this_turn=inputs['this_turn'], src_char=inputs['src_char'], tgt_out_char=inputs['tgt_out_char'], ctx_char=inputs['ctx_char'], bg_char=inputs.get('bg_char', None), neg_out_char=neg_out_char, yesno=inputs['yesno'], followup=inputs['followup'], src_text=inputs['src_text'], bg_text=inputs['bg_text'], tgt_text=inputs['tgt_text'], neg_text=neg_text, ctx_text=inputs['ctx_text'])[:2]
        else:
            sr_mean = 0
            teacher_acc = (0, 0, 0)

        if self.args['lambda2'] < 1:
            log_probs = self.model(src, src_mask, turn_ids, tgt_in, bg=bg, bg_mask=bg_mask)
            loss = self.crit(log_probs.view(-1, len(self.vocab['word2id'])), tgt_out.view(-1))
            loss_val = loss.item()
        else:
            loss_val = loss = 0
        if eval:
            return loss_val, teacher_acc, sr_mean

        # policy gradient
        if self.args['lambda2'] != 0:
            if self.args['lambda_reinforce'] != 0:
                if freeze_teacher and isinstance(teacher_loss, torch.Tensor): teacher_loss = teacher_loss.item()

                teacher_loss = teacher_loss + self.args['lambda_reinforce'] * (s_nll * (s_r - g_r)).mean()

            if freeze_teacher:
                loss = self.args['lambda2'] * teacher_loss + (1-self.args['lambda2']) * loss
            else:
                loss = teacher_loss.item() + loss

        loss_val = loss if isinstance(loss, int) or isinstance(loss, float) else loss.item()

        if self.args['lambda2'] < 1:
            loss /= 2
            loss.backward()
            if i % 2 == 1:
                if self.args['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters, self.args['max_grad_norm'])
                self.optimizer.step()
        if self.args['lambda2'] != 0 and not freeze_teacher:
            teacher_loss /= 2
            teacher_loss.backward()
            if i % 2 == 1:
                self.teacher_optimizer.step()

        return loss_val, teacher_acc, sr_mean

    def predict(self, batch, beam_size=1, return_pair_level=False, return_preds=False, return_rewards=False):
        inputs = unpack_batch(batch, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids']
        bg = inputs.get('bg', None)
        src_mask = src.eq(constant.PAD_ID)
        bg_mask = bg.eq(constant.PAD_ID) if bg is not None else None

        if not return_preds:
            self.model.eval()
        batch_size = src.size(0)
        preds = self.model.predict(src, src_mask, turn_ids, beam_size=beam_size, bg=bg, bg_mask=bg_mask, return_pair_level=return_pair_level)
        if return_preds:
            return preds
        pred_seqs = [[self.vocab['id2word'][id_] for id_ in ids] for ids in preds] # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        if not return_rewards:
            return pred_seqs

        if self.args['lambda2'] > 0:
            pred_in, pred_out, pred_out_char, pred_text = samples_to_in_out(preds, self.vocab, src.device, src.size(0))
            with torch.no_grad():
                t_r, s_r = get_reward(tgt_in, tgt_out, inputs['tgt_out_char'], inputs['tgt_text'], pred_in, pred_out, pred_out_char, pred_text, inputs, self.teacher)
        else:
            t_r = s_r = 0
        return pred_seqs, t_r, s_r

    def sample(self, batch, top_p=1, return_pair_level=False, return_preds=False):
        inputs = unpack_batch(batch, self.use_cuda)
        src, tgt_in, tgt_out, turn_ids = \
            inputs['src'], inputs['tgt_in'], inputs['tgt_out'], inputs['turn_ids']
        bg = inputs.get('bg', None)
        src_mask = src.eq(constant.PAD_ID)
        bg_mask = bg.eq(constant.PAD_ID) if bg is not None else None

        if not return_preds:
            self.model.eval()
        batch_size = src.size(0)
        preds = self.model.sample(src, src_mask, turn_ids, top_p=top_p, bg=bg, bg_mask=bg_mask, return_pair_level=return_pair_level)
        preds, nll = preds
        if return_preds:
            return preds, nll
        pred_seqs = [[self.vocab['id2word'][id_] for id_ in ids] for ids in preds] # unmap to tokens
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)

        if self.args['lambda2'] > 0:
            pred_in, pred_out, pred_out_char, pred_text = samples_to_in_out(preds, self.vocab, src.device, src.size(0))
            with torch.no_grad():
                t_r, s_r = get_reward(tgt_in, tgt_out, inputs['tgt_out_char'], inputs['tgt_text'], pred_in, pred_out, pred_out_char, pred_text, inputs, self.teacher)
        else:
            t_r = s_r = 0

        return pred_seqs, t_r, s_r

    def update_lr(self, new_lr):
        utils.change_lr(self.optimizer, new_lr)

    def save(self, filename):
        args = {k: self.args[k] for k in self.args if k != 'vocab'}
        params = {
                'model': self.model.state_dict() if self.model is not None else None,
                'config': args
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename, use_cuda=False, args=None, vocab=None):
        try:
            print(f"Loading model from '{filename}'...")
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            sys.exit(1)
        self.args = checkpoint['config']
        if args is not None:
            self.args['lambda1'] = args['lambda1']
            self.args['lambda2'] = args['lambda2']
            self.args['lambda_reinforce'] = args['lambda_reinforce']
            self.args['teacher_elmo'] = args['teacher_elmo']
            self.args['teacher_spec_only'] = args['teacher_spec_only']
            self.args['teacher_info_only'] = args['teacher_info_only']
        if vocab is not None:
            self.args['vocab'] = vocab
        self.model = Seq2SeqModel(self.args, use_cuda=use_cuda)
        self.model.load_state_dict(checkpoint['model'])

    def save_teacher(self, filename):
        params = {
                'model': filter_elmo(self.teacher.state_dict()) if self.teacher is not None else None,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load_teacher(self, filename, use_cuda=False):
        try:
            print(f"Loading answerer/teacherriminator model from '{filename}'...")
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            import pdb; pdb.set_trace()
            print("Cannot answerer/teacherriminator load model from {}".format(filename))
            sys.exit(1)
        self.teacher = TeacherModel(self.args, use_cuda=use_cuda)
        self.teacher.load_state_dict(filter_elmo(checkpoint['model']), strict=False)
