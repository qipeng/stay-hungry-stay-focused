import sys
import os
import shutil
import time
from datetime import datetime
import argparse
import json
import numpy as np
import pathlib
import pickle
from pprint import pprint
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from colors import blue, yellow, green

from models.data import QuACDataset, QuACDataLoader, QuACBatchSampler
from models.trainer import Seq2SeqTrainer as Trainer
from utils.bleu import compute_bleu
from utils.constants import MAX_TURNS, ANSST

from rouge.rouge_score import rouge_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/quac', help='Directory for all QuAC data.')
    parser.add_argument('--train_tok_file', type=str, default='train_v0.2_train.tokenized.json', help='Input file for data loader.')
    parser.add_argument('--train_idx_file', type=str, default='train_v0.2_train.idx.json', help='Input file for data loader.')
    parser.add_argument('--train_question_freq_idx_file', type=str, default='train_v0.2_train_question_freq.json', help='Input file for data loader.')
    parser.add_argument('--eval_tok_file', type=str, default='train_v0.2_val.tokenized.json', help='Input file for data loader.')
    parser.add_argument('--eval_idx_file', type=str, default='train_v0.2_val.idx.json', help='Input file for data loader.')
    parser.add_argument('--vocab_file', type=str, default='vocab.pkl', help='Vocab file.')
    parser.add_argument('--model_file', type=str, default='model.pt', help='Model file name.')
    parser.add_argument('--teacher_model_file', type=str, default='teacher_model.pt', help='Model file name.')
    parser.add_argument('--finetuned_model_file', type=str, default='finetuned_model.pt', help='Finetuned model file name.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict', 'train_teacher', 'finetune'])

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--hidden_dim_hist', type=int, default=400)
    parser.add_argument('--teacher_hidden_dim', type=int, default=200)
    parser.add_argument('--char_hidden_dim', type=int, default=100)
    parser.add_argument('--char_conv_size', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--char_emb_dim', type=int, default=20)
    parser.add_argument('--turn_emb_dim', type=int, default=10)
    parser.add_argument('--pos_emb_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--emb_dropout', type=float, default=0.3)
    parser.add_argument('--teacher_emb_dropout', type=float, default=0.2)
    parser.add_argument('--word_dropout', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--teacher_dropout', type=float, default=0.2)
    parser.add_argument('--max_dec_len', type=int, default=30)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--teacher_max_history', type=int, default=2, help="Maximum amount of conversation history the teacher's QA model uses")
    parser.add_argument('--top', type=int, default=1e10, help="Finetune only the embeddings for this many most frequent words")
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--attn_type', default='soft', choices=['soft', 'mlp', 'linear', 'deep', 'none'], help='Attention type')
    parser.add_argument('--lambda1', type=float, default=0.5, help='Mixture weight for the flow term in rewards')
    parser.add_argument('--lambda2', type=float, default=0.98, help='Weight for NCE loss')
    parser.add_argument('--lambda_reinforce', type=float, default=1, help='Weight for RL loss')
    parser.add_argument('--eval_rouge', dest='eval_ppl', action='store_false', help='Use rouge instead of perplexity as early stopping and eval metric')
    parser.add_argument('--max_turns', type=int, default=MAX_TURNS)
    parser.add_argument('--teacher_info_only', action='store_true', help='Only use/report answer novelty reward')
    parser.add_argument('--teacher_spec_only', action='store_true', help='Only use/report flow teacherriminator reward')
    parser.add_argument('--teacher_elmo', action='store_true', help='Use ELMo for input word embeddings in the teacher model')

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--decay_epoch', type=int, default=30, help="Decay the lr starting from this epoch.")
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--trainer_patience', type=int, default=10, help="If dev performance doesn't improve over this many consecutive epochs, stop training")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--model_dir', type=str, default='trained_model', help='Root dir for saving models.')
    parser.add_argument('--finetuned', action='store_true', help='Load finetuned model for evaluation')

    parser.add_argument('--output_dump', type=str, default=None, help="JSON file to dump all system predictions and references")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--debug', action='store_true', help='Debug with dev set files')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    print("Running model in {} mode".format(args['mode']))

    if args['mode'] == 'predict':
        evaluate(args)
    else:
        train(args)

def train(args):
    # load data
    print("[Loading data with batch size {}...]".format(args['batch_size']))
    writer = SummaryWriter()

    if args['debug']:
        args['train_tok_file'] = args['eval_tok_file']
        args['train_idx_file'] = args['eval_idx_file']

    dataset = QuACDataset(os.path.join(args['data_dir'], args['train_tok_file']), os.path.join(args['data_dir'], args['train_idx_file']), os.path.join(args['data_dir'], args['train_question_freq_idx_file']), max_turns=args['max_turns'])
    train_batch = QuACDataLoader(dataset, num_workers=4, batch_sampler=QuACBatchSampler(dataset, batch_size=args['batch_size']))
    with open(os.path.join(args['data_dir'], args['vocab_file']), 'rb') as f:
        vocab = pickle.load(f)
    args['vocab_size'] = len(vocab['word2id'])
    args['char_vocab_size'] = len(vocab['char2id'])
    dev_dataset = QuACDataset(os.path.join(args['data_dir'], args['eval_tok_file']), os.path.join(args['data_dir'], args['eval_idx_file']), os.path.join(args['data_dir'], args['train_question_freq_idx_file']), max_turns=args['max_turns'])
    dev_batch = QuACDataLoader(dev_dataset, batch_size=args['batch_size'], num_workers=4)

    pathlib.Path(args['model_dir']).mkdir(parents=True, exist_ok=True)
    model_file = '{}/{}'.format(args['model_dir'], args['model_file'])
    finetuned_model_file = '{}/{}'.format(args['model_dir'], args['finetuned_model_file'])
    teacher_model_file = '{}/{}'.format(args['model_dir'], args['teacher_model_file'])

    pprint(args)

    if args['mode'] == 'train':
        args['lambda2'] = 0
        args['finetune'] = False
    elif args['mode'] == 'train_teacher':
        args['lambda2'] = 1
        args['finetune'] = False
    elif args['mode'] == 'finetune':
        args['finetune'] = True

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("[Skip training because no data available...]")
        sys.exit(0)

    # start training
    args['vocab'] = vocab

    finetuning = False

    if args['finetune']:
        print('Start finetuning...')
        finetuning = True
        trainer = Trainer(model_file=model_file, use_cuda=args['cuda'], vocab=vocab, teacher_model_file=teacher_model_file, args=args)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, factor=args['lr_decay'], patience=args['patience'], mode='max')
        trainer.teacher.eval()
    else:
        lambda_rl = args['lambda_reinforce']
        args['lambda_reinforce'] = 0
        trainer = Trainer(args=args, vocab=vocab, use_cuda=args['cuda'], emb_matrix=vocab['vecs'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, factor=args['lr_decay'], patience=args['patience'], mode='min' if args['eval_ppl'] else 'max')
    print("[Training seq2seq-based question generator...]")
    global_step = 0
    max_steps = len(train_batch) * args['num_epoch']
    dev_score_history = []
    best_dev_preds = []
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}, lambda: {:.6f}'

    references = [' '.join([x.lower() for x in tgt_text]) for i in range(len(dev_dataset)) for tgt_text in dev_dataset[i]['tgt_text']]

    if args['lambda2'] > 0:
        teacher_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(trainer.teacher_optimizer, factor=args['lr_decay'], patience=args['patience'], mode='max')

    writer_tag = f"lambda1={args['lambda1']};lambda2={args['lambda2']}"
    if args['teacher_info_only']:
        writer_tag += '/informativeness_reward_only'
    elif args['teacher_spec_only']:
        writer_tag += '/specificity_reward_only'

    # eval on dev
    print("Evaluating on dev set...")
    dev_preds = []
    dev_edits = []
    dev_acc = (0, 0, 0)
    logppl = 0
    total_count = 0
    total_sent = 0
    total_reward = 0
    for i, batch in enumerate(tqdm(dev_batch)):
        if args['eval_ppl']:
            loss, acc, reward = trainer.update(batch, eval=True, freeze_teacher=args['finetune'])
            count = (batch['tgt_out'] > 0).sum().item()
            logppl += loss * count
            total_count += count
            sent_count = sum(len(x) for x in batch['tgt_text'])
            total_sent += sent_count
            dev_acc = tuple(x * sent_count + y for x, y in zip(acc, dev_acc))
            total_reward += reward * sent_count
        else:
            preds = trainer.predict(batch, args['beam_size'])
            dev_preds += preds

    dev_acc = tuple(x / total_sent for x in dev_acc)
    writer.add_scalars(writer_tag, {'spec_reward': dev_acc[0], 'novelty_reward': dev_acc[1], 'nll': dev_acc[2], 'weighted_sum': total_reward / total_sent}, 0)
    writer.flush()

    best_dev_acc = (0, 0, -1e10)
    patience = 0
    target_lambda = args['lambda2']
    finetune_start = 1
    # start training
    for epoch in range(1, args['num_epoch']+1):
        if trainer.optimizer.param_groups[0]['lr'] < args['lr'] * 1e-2 and \
            (args['lambda2'] == 0 or trainer.teacher_optimizer.param_groups[0]['lr'] < args['lr'] * 1e-2 or finetuning):
            if finetuning or lambda_rl == 0: break

            print('Start finetuning...')
            finetuning = True
            args['lambda_reinforce'] = lambda_rl
            trainer = Trainer(model_file=model_file, use_cuda=args['cuda'], vocab=vocab, teacher_model_file=teacher_model_file, args=args)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, factor=args['lr_decay'], patience=args['patience'], mode='max')
            finetune_start = epoch
            trainer.teacher.eval()
            dev_score_history = []

        train_loss = 0
        teacher_acc = (0, 0, 0)
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss, acc, reward = trainer.update(batch, eval=False, freeze_teacher=finetuning, i=global_step) # update step
            train_loss += loss
            teacher_acc = tuple(x+y for x, y in zip(acc, teacher_acc))
            if global_step % args['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
                        max_steps, epoch, args['num_epoch'], loss, duration, trainer.optimizer.param_groups[0]['lr'],
                        trainer.args['lambda2']))

        # eval on dev
        print("Evaluating on dev set...")
        dev_preds = []
        dev_edits = []
        dev_acc = (0, 0, 0)
        logppl = 0
        total_count = 0
        total_sent = 0
        total_reward = 0
        for i, batch in enumerate(tqdm(dev_batch)):
            if args['eval_ppl']:
                loss, acc, reward = trainer.update(batch, eval=True, freeze_teacher=finetuning)
                count = (batch['tgt_out'] > 0).sum().item()
                logppl += loss * count
                total_count += count
                sent_count = sum(len(x) for x in batch['tgt_text'])
                total_sent += sent_count
                dev_acc = tuple(x * sent_count + y for x, y in zip(acc, dev_acc))
                total_reward += reward * sent_count
            else:
                preds = trainer.predict(batch, args['beam_size'])
                dev_preds += preds

        if args['eval_ppl']:
            if finetuning:
                dev_score = total_reward / total_sent
            else:
                dev_score = np.exp(logppl / total_count)
        else:
            dev_score = rouge_score(references, [' '.join(x) for x in dev_preds])

        lr_scheduler.step(dev_score)

        train_loss = train_loss / len(train_batch) # avg loss per batch
        print("epoch {}: train_loss = {:.6f}, dev_score = {:.4f}".format(epoch, train_loss, dev_score))
        if args['lambda2'] != 0:
            dev_acc = tuple(x / total_sent for x in dev_acc)
            teacher_acc = tuple(x / len(train_batch) for x in teacher_acc)
            writer.add_scalars(writer_tag, {'spec_reward': dev_acc[0], 'novelty_reward': dev_acc[1], 'nll': dev_acc[2], 'weighted_sum': total_reward / total_sent}, epoch)
            writer.flush()
            teacher_lr_scheduler.step(sum(dev_acc))
            print("train_acc = {:s}, dev_acc = {:s}, teacher lr = {:.6f}".format(str(teacher_acc), str(dev_acc), trainer.teacher_optimizer.param_groups[0]['lr']))

        # save best model
        compare = (lambda new, old: new > max(old)) if finetuning else (lambda new, old: new < min(old))
        if args['lambda2'] != 1 and (len(dev_score_history) == 0 or compare(dev_score, dev_score_history)):
            if finetuning:
                trainer.save(finetuned_model_file)
            else:
                trainer.save(model_file)
            print("new best model saved.")
            best_dev_preds = dev_preds
            patience = 0

        if not finetuning and args['lambda2'] != 0 and dev_acc[0] + dev_acc[1] + 0.1 * dev_acc[2] > best_dev_acc[0] + best_dev_acc[1] + 0.1 * best_dev_acc[2]:
            trainer.save_teacher(teacher_model_file)
            print("new best teacher model saved.")
            best_dev_acc = dev_acc
            patience = 0

        dev_score_history += [dev_score]
        print("")

        if patience >= args['trainer_patience']:
            break
        patience += 1

    print("Training ended with {} epochs.".format(epoch))

    if finetuning:
        best_f, best_epoch = max(dev_score_history), np.argmax(dev_score_history)+1
        print("Best dev score = {:.2f}, at epoch = {}".format(best_f, best_epoch))
    else:
        best_f, best_epoch = min(dev_score_history), np.argmin(dev_score_history)+1
        print("Best dev perplexity = {:.2f}, at epoch = {}".format(best_f, best_epoch))

def evaluate(args):
    # file paths
    model_file = '{}/{}'.format(args['model_dir'], args['model_file'])
    finetuned_model_file = '{}/{}'.format(args['model_dir'], args['finetuned_model_file'])
    teacher_model_file = '{}/{}'.format(args['model_dir'], args['teacher_model_file'])

    # load model
    with open(os.path.join(args['data_dir'], args['vocab_file']), 'rb') as f:
        vocab = pickle.load(f)
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(model_file=finetuned_model_file if args['finetuned'] else model_file, use_cuda=use_cuda, vocab=vocab, teacher_model_file=teacher_model_file, args=args)
    if hasattr(trainer, 'model') and trainer.model is not None:
        trainer.model.eval()
    if hasattr(trainer, 'teacher') and trainer.teacher is not None:
        trainer.teacher.eval()
    loaded_args = trainer.args
    trainer.args['vocab'] = vocab

    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['mode']:
            loaded_args[k] = args[k]

    # laod data
    print("Loading data with batch size {}...".format(args['batch_size']))
    dataset = QuACDataset(os.path.join(args['data_dir'], args['eval_tok_file']), os.path.join(args['data_dir'], args['eval_idx_file']), os.path.join(args['data_dir'], args['train_question_freq_idx_file']), max_turns=args['max_turns'])
    batch = QuACDataLoader(dataset, batch_size=args['batch_size'], num_workers=4)

    # skip eval if dev data does not exist
    if len(batch) == 0:
        print("Skip evaluation because no dev data is available...")
        print("Lemma score:")
        print("{} ".format(args['lang']))
        sys.exit(0)

    #references = [[[x.lower() for x in dataset[i]['tgt_out_text'][:-1]]] for i in range(len(dataset))]
    references = [' '.join([x.lower() for x in tgt_text]) for i in range(len(dataset)) for tgt_text in dataset[i]['tgt_text']]
    print("Running the seq2seq model...")
    preds = [[] for _ in range(args['num_samples'])]
    edits = []
    offset = 0
    lastsrc = None
    logppl = 0
    total_count = 0
    total_reward = 0
    total_tgt_reward = 0
    total_sent = 0
    unique_questions = []
    ref_unique_questions = []

    dumped_data = []

    for i, b in enumerate(batch):
        #ps = trainer.predict(b, args['beam_size'])
        pall = []
        pls = []
        batch_size = len(b['ctx_text'])
        # import pdb; pdb.set_trace()
        for s in range(args['num_samples']):
            if args['eval_ppl']:
                loss, acc, _ = trainer.update(b, eval=True)
                count = (b['tgt_out'] > 0).sum().item()
                logppl += loss * count
                total_count += count
            pl = None
            #if args['top_p'] < 1:
            #    #ps, pl = trainer.sample(b, top_p=args['top_p'], return_pair_level=True)
            #    ps, tgt_reward, pred_reward = trainer.sample(b, top_p=args['top_p'], return_pair_level=True)
            #else:
                #ps, pl = trainer.predict(b, args['beam_size'], return_pair_level=True)
            ps, tgt_reward, pred_reward = trainer.predict(b, args['beam_size'], return_pair_level=True, return_rewards=True)

            sent_count = (b['tgt_out'] > 0).any(2).sum().item()
            total_sent += sent_count
            total_reward += pred_reward.sum().item()
            total_tgt_reward += tgt_reward.sum().item()

            offset = 0
            ps_final = []
            tgt_reward = tgt_reward.view(batch_size, -1)
            pred_reward = pred_reward.view(batch_size, -1)
            for j in range(batch_size):
                pred = ps[offset:offset+len(b['src_text'][j])]

                ps_final.extend(pred)
                offset += len(ps) // batch_size

                ref_questions = [' '.join(x) for x in b['tgt_text'][j]]
                pred_questions = [' '.join(x) for x in pred]
                unique_questions.append(sum([1 for i, x in enumerate(pred_questions) if x not in ref_questions[:i]]))
                ref_unique_questions.append(sum([1 for i, x in enumerate(ref_questions) if x not in ref_questions[:i]]))

            ps = ps_final

            preds[s] += ps
            pall.append(ps)
            if pl is not None:
                pls.append(pl.squeeze(1))

        offset = 0
        for j in range(batch_size):
            instance = {}
            instance['bg'] = ' '.join(b['bg_text'][j])
            instance['section_title'] = ' '.join(b['src_text'][j][0])
            instance['qas'] = []
            num_questions = len(b['src_text'][j])
            print("=" * 20)
            print(f"{' '.join(b['bg_text'][j])}\t{' '.join(b['src_text'][j][0])}\t{' '.join(b['tgt_text'][j][0]).lower()}\t{' '.join(ps[offset])}")
            print(f"Metrics: {tgt_reward[j][0]} {pred_reward[j][0]}")

            qa = {'question_ref': ' '.join(b['tgt_text'][j][0]).lower(), 'question': ' '.join(ps[offset])}

            for k in range(1, num_questions):
                idx = b['src_text'][j][k].index(ANSST)
                ans = b['src_text'][j][k][idx:]
                qa['answer'] = ' '.join(ans)
                instance['qas'].append(qa)
                print(f"\t{' '.join(ans)}\t{' '.join(b['tgt_text'][j][k]).lower()}\t{' '.join(ps[offset+k])}")
                print(f"Metrics: {tgt_reward[j][k]} {pred_reward[j][k]}")
                qa = {'question_ref': ' '.join(b['tgt_text'][j][k]).lower(), 'question': ' '.join(ps[offset+k])}

            # add last question
            instance['qas'].append(qa)

            offset += num_questions

            dumped_data.append(instance)

    if args['eval_ppl']:
        score = np.exp(logppl / total_count)
        print("Perplexity:", score)

    score = rouge_score([x for _ in range(args['num_samples']) for x in references], [' '.join(x) for p in preds for x in p])
    print('ROUGE-L:', score * 100)
    print('Reward (pred/ref):', total_reward / total_sent, total_tgt_reward / total_sent)
    print('Unique questions (pred/ref/total_sent):', len(set([' '.join(x) for x in preds[0]])), len(set(references)), total_sent)
    print('Unique questions per dialogue (pred/ref):', np.mean(unique_questions), np.mean(ref_unique_questions))

    from collections import Counter

    print(sorted(Counter(Counter(references).values()).items()))
    print(sorted(Counter(Counter([' '.join(p) for p in preds[0]]).values()).items()))

    if args['output_dump'] is not None:
        with open(args['output_dump'], 'w') as f:
            json.dump(dumped_data, f)

if __name__ == '__main__':
    main()
