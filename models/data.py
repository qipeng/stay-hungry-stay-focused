"""
This file contains the data pipeline for the models.
"""

from allennlp.commands.elmo import ElmoEmbedder
from collections import defaultdict
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from utils.constants import *

MAX_SKETCHES = 5

def pad_lists(x, fill_val=0, dtype=np.int64):
    size = [len(x)]
    y = x
    while isinstance(y[0], list) or isinstance(y[0], np.ndarray):
        yy = []
        mx = 0
        for t in y:
            mx = max(len(t), mx)
            yy.extend(t)
        size.append(mx)
        y = yy

    res = np.full(size, fill_val, dtype=dtype)
    assert len(size) <= 4
    if len(size) == 1:
        res = np.array(x, dtype=dtype)
    elif len(size) == 2:
        for i in range(len(x)):
            res[i, :len(x[i])] = x[i]
    elif len(size) == 3:
        for i in range(len(x)):
            for j in range(len(x[i])):
                res[i, j, :len(x[i][j])] = x[i][j]
    elif len(size) == 4:
        for i in range(len(x)):
            for j in range(len(x[i])):
                for k in range(len(x[i][j])):
                    res[i, j, k, :len(x[i][j][k])] = x[i][j][k]

    return res

def pad_char_start_end(char_ids):
    return [[CHAR_START_ID] + w + [CHAR_END_ID] for w in char_ids]

class QuACDataset(Dataset):
    '''
    Dataset to load QuAC for question generation.
    '''
    def __init__(self, tok_file, idx_file, freq_idx_file, max_turns=-1, freq_cutoff=1):
        with open(tok_file) as f:
            tok_data = json.load(f)

        with open(idx_file) as f:
            idx_data = json.load(f)

        with open(freq_idx_file) as f:
            freq_idx_data = json.load(f)

        self.max_turns = max_turns
        self.freq_cutoff = freq_cutoff
        self.preprocess(tok_data, idx_data, freq_idx_data)

    def preprocess(self, tok_data, idx_data, freq_idx_data):
        self.freq_questions = []
        self.freq_q_prob = []
        for (itgt, ctgt, ttgt, tgt_idf), freq in freq_idx_data:
            if freq <= self.freq_cutoff: continue
            self.freq_questions.append((itgt, pad_char_start_end(ctgt), ttgt, tgt_idf))
            self.freq_q_prob.append(freq)
        sum_q_freq = sum(self.freq_q_prob)
        self.freq_q_prob = [x / sum_q_freq for x in self.freq_q_prob]

        self.data = []

        for tpara, ipara in tqdm(zip(tok_data, idx_data), total=len(tok_data)):
            tsrc = [TITLEST] + tpara['title'] + [TITLEEN, BGST] + tpara['background'][:MAX_BACKGROUND] + [BGEN]
            isrc = [TITLEST_ID] + ipara['title'] + [TITLEEN_ID, BGST_ID] + ipara['background'][:MAX_BACKGROUND] + [BGEN_ID]
            csrc = pad_char_start_end([[TITLEST_ID]] + ipara['title_char'] + [[TITLEEN_ID], [BGST_ID]] + ipara['background_char'][:MAX_BACKGROUND] + [[BGEN_ID]])

            tbg = [] + tsrc
            ibg = [] + isrc
            cbg = [] + csrc

            turn_ids = [-1] * len(tsrc)

            # clear src variables for dialogue history
            tsrc = []
            isrc = []
            csrc = []
            turn_ids = []

            tsrc += [SECST] + tpara['section_title'] + [SECEN]
            isrc += [SECST_ID] + ipara['section_title'] + [SECEN_ID]
            csrc += pad_char_start_end([[SECST_ID]] + ipara['section_title_char'] + [[SECEN_ID]])
            src_idf = [0] + ipara['section_title_idf'] + [0]
            turn_ids += [0] * (len(ipara['section_title']) + 2)

            ans_mask = np.zeros((len(ipara['context'][:MAX_CONTEXT]), MAX_TURNS), dtype=np.int64)

            this_para = defaultdict(list)
            for turnid, (tqa, iqa) in enumerate(zip(tpara['qas'], ipara['qas'])):
                ttgt_in = [SOS] + tqa['question']
                ttgt_out = tqa['question'] + [EOS]
                itgt_in = np.array([SOS_ID] + iqa['question'], dtype=np.int64)
                itgt_out = np.array(iqa['question'] + [EOS_ID], dtype=np.int64)
                ctgt_in = pad_char_start_end([[SOS_ID]] + iqa['question_char'])
                ctgt_out = pad_char_start_end(iqa['question_char'] + [[EOS_ID]])

                this_para['src_text'].append([] + tsrc)
                this_para['src_idx'].append(pad_lists(isrc, dtype=np.int64))
                this_para['src_char'].append(pad_lists(csrc, dtype=np.int64))
                this_para['src_idf'].append(pad_lists(src_idf, dtype=np.float32))
                this_para['tgt_text'].append([x.lower() for x in tqa['question']])
                this_para['tgt_in_idx'].append(pad_lists(itgt_in, dtype=np.int64))
                this_para['tgt_in_char'].append(pad_lists(ctgt_in, dtype=np.int64))
                this_para['tgt_out_idx'].append(pad_lists(itgt_out, dtype=np.int64))
                this_para['tgt_out_char'].append(pad_lists(ctgt_out, dtype=np.int64))
                this_para['turn_ids'].append(pad_lists(turn_ids, fill_val=-1, dtype=np.int64))
                this_para['start'].append(iqa['start'])
                this_para['end'].append(iqa['end'])
                this_para['yesno'].append(iqa['yesno'])
                this_para['followup'].append(iqa['followup'])
                this_para['ans_mask'].append(np.array(ans_mask))
                this_para['this_turnid'].append(turnid)

                ans_mask[iqa['start']:iqa['end'], turnid] += 1 # in span
                ans_mask[iqa['start']:iqa['start']+1, turnid] += 1 # beginning of span
                ans_mask[iqa['end']-1:iqa['end'], turnid] += 2 # end of span

                # append Q and A with separators
                tsrc = [QUESST] + [x.lower() for x in tqa['question']] + [QUESEN, ANSST] + tqa['answer'] + [ANSEN]
                isrc = [QUESST_ID] + iqa['question'] + [QUESEN_ID, ANSST_ID] + iqa['answer'] + [ANSEN_ID]
                csrc = pad_char_start_end([[QUESST_ID]] + iqa['question_char'] + [[QUESEN_ID], [ANSST_ID]] + iqa['answer_char'] + [[ANSEN_ID]])
                turn_ids = [turnid + 1] * (len(tqa['question']) + len(tqa['answer']) + 4)

                ques_count = sum(1 for x in isrc if x == QUESST_ID) - 1
                if self.max_turns >= 0 and ques_count > self.max_turns:
                    idx = len(isrc) - 1
                    count = 0
                    while idx > 0:
                        if isrc[idx] == QUESST_ID:
                            count += 1
                            if count > self.max_turns:
                                break
                        idx -= 1
                    tsrc = tsrc[idx:]
                    isrc = isrc[idx:]
                    csrc = csrc[idx:]
                    turn_ids = turn_ids[idx:]

            datum = dict()
            datum['ctx_text'] = tpara['context'][:MAX_CONTEXT]
            datum['ctx_idx'] = ipara['context'][:MAX_CONTEXT]
            datum['ctx_char'] = pad_lists(pad_char_start_end(ipara['context_char'][:MAX_CONTEXT]), dtype=np.int64)

            datum['bg_text'] = tbg
            datum['bg_idx'] = ibg
            datum['bg_char'] = cbg

            for k in ['src_text', 'tgt_text', 'start', 'end', 'yesno', 'followup', 'this_turnid']:
                datum[k] = this_para[k]

            for k in ['src_idx', 'src_char', 'tgt_in_idx', 'tgt_in_char', 'tgt_out_idx', 'tgt_out_char', 'ans_mask']:
                datum[k] = pad_lists(this_para[k], dtype=np.int64)

            datum['turn_ids'] = pad_lists(this_para['turn_ids'], fill_val=-1, dtype=np.int64)

            self.data.append(datum)

    def __getitem__(self, i):
        res = self.data[i]
        idx = np.random.choice(len(self.freq_questions), size=(len(res['tgt_text']),), p=self.freq_q_prob)
        res['neg_out_idx'] = pad_lists([self.freq_questions[i][0] for i in idx])
        res['neg_out_char'] = pad_lists([self.freq_questions[i][1] for i in idx])
        res['neg_text'] = [self.freq_questions[i][2] for i in idx]
        return res

    def __len__(self):
        return len(self.data)

class QuACBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, noise=.1):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.noisy_val = lambda x: x * (1 - noise + random.random() * noise * 2)
        self.reset_batches_idx()

    def reset_batches_idx(self):
        idx = sorted([(self.noisy_val(len(x['tgt_text'])), self.noisy_val(len(x['ctx_text'])),
            i) for i, x in enumerate(self.dataset)], reverse=True)
        batches = [[x[-1] for x in idx[st:st+self.batch_size]] for st in range(0, len(self.dataset), self.batch_size)]
        self.batches = batches
        random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches
        self.reset_batches_idx()

    def __len__(self):
        return len(self.batches)

class QuACDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn

        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        src = torch.from_numpy(pad_lists([x['src_idx'] for x in batch_data]))
        src_char = torch.from_numpy(pad_lists([x['src_char'] for x in batch_data]))
        ctx = torch.from_numpy(pad_lists([x['ctx_idx'] for x in batch_data]))
        ctx_char = torch.from_numpy(pad_lists([x['ctx_char'] for x in batch_data]))
        tgt_in = torch.from_numpy(pad_lists([x['tgt_in_idx'] for x in batch_data]))
        tgt_in_char = torch.from_numpy(pad_lists([x['tgt_in_char'] for x in batch_data]))
        tgt_out = torch.from_numpy(pad_lists([x['tgt_out_idx'] for x in batch_data]))
        tgt_out_char = torch.from_numpy(pad_lists([x['tgt_out_char'] for x in batch_data]))
        neg_out = torch.from_numpy(pad_lists([x['neg_out_idx'] for x in batch_data]))
        neg_out_char = torch.from_numpy(pad_lists([x['neg_out_char'] for x in batch_data]))

        this_turn = torch.from_numpy(pad_lists([x['this_turnid'] for x in batch_data]))
        ans_mask = torch.from_numpy(pad_lists([x['ans_mask'] for x in batch_data]))

        turn_ids = torch.from_numpy(pad_lists([x['turn_ids'] for x in batch_data], fill_val=-1))
        start = torch.from_numpy(pad_lists([x['start'] for x in batch_data], fill_val=-1))
        end = torch.from_numpy(pad_lists([x['end'] for x in batch_data], fill_val=-1))
        yesno = torch.from_numpy(pad_lists([x['yesno'] for x in batch_data], fill_val=-1))
        followup = torch.from_numpy(pad_lists([x['followup'] for x in batch_data], fill_val=-1))

        retval = {'src': src,
                  'src_char': src_char,
                  'src_text': [x['src_text'] for x in batch_data],
                  'tgt_in': tgt_in,
                  'tgt_out': tgt_out,
                  'tgt_out_char': tgt_out_char,
                  'tgt_text': [x['tgt_text'] for x in batch_data],
                  'neg_out': neg_out,
                  'neg_out_char': neg_out_char,
                  'neg_text': [x['neg_text'] for x in batch_data],
                  'turn_ids': turn_ids,
                  'ctx': ctx,
                  'ctx_char': ctx_char,
                  'ctx_text': [x['ctx_text'] for x in batch_data],
                  'start': start,
                  'end': end,
                  'yesno': yesno,
                  'followup': followup,
                  'this_turn': this_turn,
                  'ans_mask': ans_mask}
        if 'bg_text' in batch_data[0]:
            bg = torch.from_numpy(pad_lists([x['bg_idx'] for x in batch_data]))
            bg_char = torch.from_numpy(pad_lists([x['bg_char'] for x in batch_data]))
            retval['bg'] = bg
            retval['bg_char'] = bg_char
            retval['bg_text'] = [x['bg_text'] for x in batch_data]
        return retval

if __name__ == "__main__":
    dataset = QuACDataset('data/quac/val_v0.2.tokenized.json', 'data/quac/val_v0.2.idx.json')
    loader = QuACDataLoader(dataset, batch_size=5)

    for batch in loader:
        print(batch)
        break
