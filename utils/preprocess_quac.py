from allennlp.commands.elmo import ElmoEmbedder
from argparse import ArgumentParser
from collections import Counter
from joblib import Parallel, delayed
import json
from multiprocessing import Pool
import numpy as np
import os
import pickle
import random
import re
from tqdm import tqdm

from utils.spacy import bulk_tokenize
from utils.constants import VOCAB_PREFIX, UNK_ID, UNK, FOLLOWUP_TO_ID, YESNO_TO_ID

def parsed_to_word_tuple(question, answer):
    f = lambda w: UNK if w is None else w
    tpl = parsed_to_tuple(question, answer)
    return (f(tpl[1]), f(tpl[3]), f(tpl[5]), f(tpl[7]))

def split_symbol(sent, offsets, symbol, test=lambda splitted, w: True):
    res = []
    reso = []
    for w, o in zip(sent, offsets):
        splitted = [w1 for w1 in w.split(symbol)]
        if len(splitted) > 1 and test(splitted, w):
            st = o[0]
            if len(splitted[0]) > 0:
                res.append(splitted[0])
                reso.append((st, st+len(splitted[0])))
                st += len(splitted[0])
            for w1 in splitted[1:]:
                res.append(symbol)
                reso.append((st, st+1))
                st += 1
                if len(w1) > 0:
                    res.append(w1)
                    reso.append((st, st+len(w1)))
                    st += len(w1)
        else:
            res.append(w)
            reso.append(o)
    return res, reso

def split_common_symbols(sent, offsets):
    sent, offsets = split_symbol(sent, offsets, '/', lambda splitted, w: not w.startswith('</'))
    sent, offsets = split_symbol(sent, offsets, '.', lambda splitted, w: all(len(x) > 3 or len(x) == 0 for x in splitted) and not re.match('^[0-9\.]+$', w))
    sent, offsets = split_symbol(sent, offsets, '-')
    return sent, offsets

def tokenize_one(item):
    strings_to_tokenize = [item['title'], item['section_title'], item['paragraphs'][0]['context'], item['background']]
    for qa in item['paragraphs'][0]['qas']:
        strings_to_tokenize.append(qa['question'])
        strings_to_tokenize.append(qa['orig_answer']['text'])

    tokenized, offsets = bulk_tokenize(strings_to_tokenize, return_offsets=True)

    #tokenized, offsets = list(map(list, zip(*[split_common_symbols(sent, o) for sent, o in zip(tokenized, offsets)])))

    retval = {'title': tokenized[0], 'section_title': tokenized[1], 'context': tokenized[2], 'background': tokenized[3] }
    tokenized = tokenized[4:]
    ctx_offsets = [(st-offsets[2][0][0], en-offsets[2][0][0]) for st, en in offsets[2]]

    qas = []
    parsed_idx = 0
    for qa in item['paragraphs'][0]['qas']:
        ans = tokenized[1]
        if qa['yesno'] == 'y':
            ans = ['Yes', ','] + tokenized[1]
        elif qa['yesno'] == 'n':
            ans = ['No', ','] + tokenized[1]

        char_st = qa['orig_answer']['answer_start']
        char_en = char_st + len(qa['orig_answer']['text'])
        ans_st = -1
        ans_en = -1
        for idx, (st, en) in enumerate(ctx_offsets):
            if en > char_st and ans_st < 0:
                ans_st = idx
            if st >= char_en and ans_en < 0:
                ans_en = idx
        if ans_en < 0:
            ans_en = len(ctx_offsets)
        assert ''.join(tokenized[1]) in ''.join(retval['context'][ans_st:ans_en]), '{} {}'.format(str(retval['context'][ans_st:ans_en]), str(tokenized[1]))
        qas.append({'question': tokenized[0], 'answer': ans, 'id': qa['id'],
            'start': ans_st, 'end': ans_en, 'yesno': qa['yesno'], 'followup': qa['followup']})
        tokenized = tokenized[2:]
        offsets = offsets[2:]
        parsed_idx += 2

    retval['qas'] = qas

    return retval

pool = Pool()

def tokenize_data(data):
    print('Tokenizing...')
    return list(tqdm(pool.imap(tokenize_one, data), total=len(data)))

def prepare_vocab(tokenized_data, vocab_file, wordvec_file, wordvec_dim, min_freq=3):
    print('Loading word vectors...')
    words = [] + VOCAB_PREFIX
    vecs = [np.random.randn(wordvec_dim) for _ in VOCAB_PREFIX]
    vecs[0] *= 0
    word2id = {w:i for i, w in enumerate(words)}

    print('Counting word frequency in the training set...')
    doc_freq = Counter()
    data_words = Counter()
    for item in tokenized_data:
        doc_words = set([x.lower() for x in item['title'] + item['section_title'] + item['background'] + item['context']])
        data_words.update([x for x in item['title']])
        data_words.update([x for x in item['section_title']])
        data_words.update([x for x in item['background']])
        data_words.update([x for x in item['context']])
        for qa in item['qas']:
            data_words.update([x.lower() for x in qa['question']])
            data_words.update([x for x in qa['answer']])
            doc_words.update([x.lower() for x in qa['question'] + qa['answer']])

        doc_freq.update(doc_words)

    data_chars = Counter(c for w in data_words for c in w)
    data_words_final = Counter()
    for w in data_words:
        data_words_final[w.lower()] += data_words[w]
    data_words = data_words_final
    for w in list(data_words.keys()):
        if w in word2id or data_words[w] < min_freq:
            del data_words[w]

    if os.path.exists(wordvec_file + '.pkl'):
        with open(wordvec_file + '.pkl', 'rb') as f:
            pretrained_words0 = pickle.load(f)
            pretrained_vecs0 = pickle.load(f)

        pretrained_words = []
        pretrained_vecs = []
        for word, vec in zip(pretrained_words0, pretrained_vecs0):
            if word.lower() in data_words:
                pretrained_words.append(word)
                pretrained_vecs.append(vec)

        pretrained_words_set = set(pretrained_words)
    else:
        pretrained_words = []
        pretrained_vecs = []
        pretrained_words_set = set()
        with open(wordvec_file) as f:
            processed_lines = 0
            for line in f:
                line = line.rstrip().split(' ')
                vec = [float(x) for x in line[-wordvec_dim:]]
                word = ' '.join(line[:-wordvec_dim])

                if word == '<unk>':
                    vecs[UNK_ID] = vec
                elif word.lower() not in pretrained_words_set and word.lower() in data_words:
                    pretrained_words.append(word.lower())
                    pretrained_words_set.add(word.lower())
                    pretrained_vecs.append(vec)

                processed_lines += 1

        with open(wordvec_file + '.pkl', 'wb') as f:
            pickle.dump(pretrained_words, f)
            pickle.dump(pretrained_vecs, f)

    print(f'{len(pretrained_words)} words loaded from the word vectors file.')

    #for w in data_words:
    #    if w in pretrained_words_set: continue
    #    words.append(w)
    #    vecs.append(np.random.randn(wordvec_dim))
    #    word2id[w] = len(word2id)
    words.append('cannotanswer')
    vecs.append(np.random.randn(wordvec_dim))
    word2id['cannotanswer'] = len(word2id)

    for w, vec in zip(pretrained_words, pretrained_vecs):
        if w not in data_words: continue
        words.append(w)
        vecs.append(vec)
        word2id[w] = len(word2id)

    vecs = np.array(vecs, dtype=np.float32)

    id2char = [] + VOCAB_PREFIX
    char2id = {c:i for i, c in enumerate(id2char)}
    for i, c in enumerate(data_chars.keys()):
        char2id[c] = len(id2char)
        id2char.append(c)

    assert len(word2id) == len(words) == vecs.shape[0]
    vocab = {'word2id': word2id, 'id2word': words, 'vecs': vecs, 'char2id': char2id, 'id2char': id2char}

    wordid2chars = []
    for i, w in enumerate(words):
        if w in VOCAB_PREFIX:
            wordid2chars.append([char2id[w]])
        else:
            wordid2chars.append([char2id[c] for c in w])
    vocab['wordid2chars'] = wordid2chars

    wordid2docfreq = []
    for i, w in enumerate(words):
        if w in VOCAB_PREFIX:
            if w != UNK:
                wordid2docfreq.append(np.inf)
            else:
                wordid2docfreq.append(1)
        else:
            wordid2docfreq.append(doc_freq[w])
    vocab['wordid2docfreq'] = wordid2docfreq

    print(f'{len(word2id) - len(pretrained_words) - len(VOCAB_PREFIX)} words added from the training set. Total vocab size: {len(word2id)}')

    return vocab

def map_data(tokenized_data, vocab):
    def map_field(field):
        return [vocab['word2id'].get(x.lower(), UNK_ID) for x in field]

    def map_char(field, do_lower=False):
        if do_lower:
            return [[vocab['char2id'].get(c, UNK_ID) for c in w.lower()] if w not in VOCAB_PREFIX else [vocab['char2id'][w]] for w in field]
        else:
            return [[vocab['char2id'].get(c, UNK_ID) for c in w] if w not in VOCAB_PREFIX else [vocab['char2id'][w]] for w in field]

    def map_idf(field):
        return [1 / vocab['wordid2docfreq'][vocab['word2id'].get(x.lower(), UNK_ID)] for x in field]

    def copy_mask(src, dst):
        return [[1 if w1.lower() == w2.lower() else 0 for w2 in dst] for w1 in src]

    def map_one(item):
        retval = {'title': map_field(item['title']),
                'title_char': map_char(item['title']),
                'section_title': map_field(item['section_title']),
                'section_title_idf': map_idf(item['section_title']),
                'section_title_char': map_char(item['section_title']),
                'background': map_field(item['background']),
                'background_char': map_char(item['background']),
                'context': map_field(item['context']),
                'context_char': map_char(item['context']),
                'qas': [{'question': map_field(x['question']), 'answer': map_field(x['answer']),
                    'question_char': map_char(x['question'], do_lower=True), 'answer_char': map_char(x['answer']),
                    'question_idf': map_idf(x['question']), 'answer_idf': map_idf(x['answer']),
                    'start': x['start'],
                    'end': x['end'],
                    'followup': FOLLOWUP_TO_ID[x['followup']],
                    'yesno': YESNO_TO_ID[x['yesno']]} for x in item['qas']]}
        return retval

    #elmo = ElmoEmbedder(cuda_device=0)
    #print('Computing ELMo features...')
    #elmo_features = [x[2] for x in tqdm(elmo.embed_sentences([item['context'] for item in tokenized_data], 20), total=len(tokenized_data))]

    print('Mapping tokenized data to indices...')
    return Parallel(n_jobs=-1, backend="threading")(delayed(map_one)(item) for item in tqdm(tokenized_data))

def split_train_data(quac_dir, file_name, val_count, tokenized_data, mapped_data):
    print('splitting training data into train_train and train_val...')

    # count wikipedia title occurrences
    title_counter = Counter([' '.join(x['title']) for x in tokenized_data])
    total_sections = sum(title_counter.values())
    train_sections = total_sections - val_count

    # split data by wikipedia articles to minimize leakage
    titles = list(title_counter.keys())
    random.seed(31415) # make sure we always get the same split
    random.shuffle(titles)
    train_sec = 0
    train_titles = set()
    for t in titles:
        train_titles.add(t)
        train_sec += title_counter[t]
        if train_sec >= train_sections:
            break

    train_tok = []
    train_idx = []
    val_tok = []
    val_idx = []

    print(f'Split into {train_sec} training sections and {total_sections - train_sec} validation sections.')

    for tok, idx in zip(tokenized_data, mapped_data):
        if ' '.join(tok['title']) in train_titles:
            train_tok.append(tok)
            train_idx.append(idx)
        else:
            val_tok.append(tok)
            val_idx.append(idx)

    def write_to_file(split, tok, idx):
        tok_file = os.path.join(args.quac_dir, f'_{split}.tokenized'.join(os.path.splitext(args.file_name)))
        idx_file = os.path.join(args.quac_dir, f'_{split}.idx'.join(os.path.splitext(args.file_name)))
        with open(tok_file, 'w') as f:
            json.dump(tok, f)
        with open(idx_file, 'w') as f:
            json.dump(idx, f)

    print('writing to file...')
    write_to_file('train', train_tok, train_idx)
    write_to_file('val', val_tok, val_idx)

    print('counting question frequency in the training set...')
    train_questions = [(tuple(iqa['question']), tuple(tuple(w) for w in iqa['question_char']), tuple([w.lower() for w in tqa['question']]), tuple(iqa['question_idf'])) for tpara, ipara in zip(train_tok, train_idx) for tqa, iqa in zip(tpara['qas'], ipara['qas'])]
    q_counter = Counter(train_questions)
    print('saving training question frequency to file...')
    with open(os.path.join(args.quac_dir, '_train_question_freq'.join(os.path.splitext(args.file_name))), 'w') as f:
        json.dump([[k, q_counter[k]] for k in q_counter], f)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--quac_dir', default='data/quac', help="Data directory for QuAC, should contain train_v0.2.json and val_v0.2.json")
    parser.add_argument('--file_name', default='train_v0.2.json', help="Data file to process")
    parser.add_argument('--wordvec_file', default='data/glove/glove.6B.100d.txt', help="File containing pretrained word embeddings")
    parser.add_argument('--vocab_file', default='vocab.pkl', help="Vocab file to save or load vocab from")
    parser.add_argument('--eval', action='store_true', help="Whether we are processing the dev/test split")
    parser.add_argument('--wordvec_dim', default=100, type=int, help="Dimension of word embeddings")
    parser.add_argument('--min_freq', default=3, type=int, help="Words in the training set occurring less than this many times will not get its own embedding")

    args = parser.parse_args()

    with open(os.path.join(args.quac_dir, args.file_name)) as f:
        data = json.load(f)

    tokenized_file = os.path.join(args.quac_dir, '.tokenized'.join(os.path.splitext(args.file_name)))
    if os.path.exists(tokenized_file):
        with open(tokenized_file) as f:
            tokenized_data = json.load(f)
    else:
        tokenized_data = tokenize_data(data['data'])
        with open(tokenized_file, 'w') as f:
            json.dump(tokenized_data, f)

    vocab_file = os.path.join(args.quac_dir, args.vocab_file)
    if not args.eval and not os.path.exists(vocab_file):
        vocab = prepare_vocab(tokenized_data, vocab_file, args.wordvec_file, args.wordvec_dim, min_freq=args.min_freq)
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
    else:
        if not os.path.exists(vocab_file):
            print('[ERROR] To preprocess evaluation data, you need to preprocess the training data first to obtain the vocabulary.')
            exit()
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)

    mapped_file = os.path.join(args.quac_dir, '.idx'.join(os.path.splitext(args.file_name)))
    if not os.path.exists(mapped_file):
        mapped_data = map_data(tokenized_data, vocab)
        with open(mapped_file, 'w') as f:
            json.dump(mapped_data, f)
    else:
        with open(mapped_file, 'r') as f:
            mapped_data = json.load(f)

    if not args.eval:
        split_train_data(args.quac_dir, args.file_name, 1000, tokenized_data, mapped_data)
