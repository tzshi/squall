import argparse

import json
import random
import time
import os
import torch
import numpy as np
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

from torch.nn.utils import clip_grad_norm_
from model import TableParser

from dataset import load_dataset, em_process
import dataset
from vocab import build_vocab

from evaluator import Evaluator
import logging


logger = logging.getLogger()



def load_words(dataset, cutoff=1):

    _vocab = build_vocab(dataset, cutoff)

    return _vocab


def evaluate(data_loader, model, evaluator, gold_decode=False):
    lf_accu = 0
    all_accu = 0
    total = 0
    all_preds = list()
    log_probs = []

    for idx, batch in enumerate(tqdm(data_loader)):
        model.eval()
        prediction, _log_probs = model(batch, isTrain=False, gold_decode=gold_decode)
        ex_acc = evaluator.evaluate(prediction)
        log_probs.extend(_log_probs)
        for d in prediction:
            total += 1
            if d['result'][0]['sql'] == d['result'][0]['tgt']:
                lf_accu += 1
        if ex_acc == 1:
            prediction[0]['correct'] = 1
        else:
            prediction[0]['correct'] = 0
        all_preds.extend(prediction)
        all_accu += ex_acc

    perplexity = 2 ** (-np.average(log_probs) / np.log(2.))
    logger.info('logical form accurate: {}/{} = {}%'.format(lf_accu, total, lf_accu / total * 100))
    logger.info('num of execution correct: {}/{} = {}%'.format(all_accu, total, all_accu / total * 100))
    logger.info('perplexity: {}'.format(perplexity))
    if gold_decode:
        return -perplexity, all_preds
    else:
        return all_accu, all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Table Parse')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--data-workers', type=int, default=2)
    parser.add_argument('--train-file', type=str, default='../data/train-0.json')
    parser.add_argument('--dev-file', type=str, default='../data/dev-0.json')
    parser.add_argument('--test-file', type=str, default='../data/wtq-test.json')
    parser.add_argument('--pred-file', type=str, default='../log/pred.json')
    parser.add_argument('--embed-file', type=str, default='../embeddings/glove.100d.training.json')

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--input-size', type=int, default=300)
    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.2)

    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--save-model', type=str, default='../log/tb_rank.pt')
    parser.add_argument('--load-model', type=str, default='../log/tb_rank.pt')
    parser.add_argument('--log-file', type=str, default = '../log/log_file_rank.log')

    parser.add_argument('--dec-loss', action='store_true', default=False)
    parser.add_argument('--enc-loss', action='store_true', default=False)
    parser.add_argument('--aux-col', action='store_true', default=False)
    parser.add_argument('--gold-decode', action='store_true', default=False)
    parser.add_argument('--gold-attn', action='store_true', default=False)
    parser.add_argument('--sample', type=int, default=50000)
    parser.add_argument('--bert', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(args.log_file, 'a')

    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    ## Add exact match feature (em_process)
    train_exs = em_process(load_dataset(args.train_file))
    if args.sample != 0:
        train_exs = train_exs[:args.sample]

    if args.dev_file is None:
        dev_exs = train_exs[-500:]
        train_exs = train_exs[:-500]
    else:
        dev_exs = em_process(load_dataset(args.dev_file))


    vocab = load_words(train_exs, cutoff=5)

    device = torch.device("cuda" if args.cuda else "cpu")

    train_dataset = dataset.WikiTableDataset(train_exs, vocab)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    dev_dataset = dataset.WikiTableDataset(dev_exs, vocab)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=1,
                                               sampler=dev_sampler, num_workers=args.data_workers,
                                               collate_fn=dataset.batchify, pin_memory=args.cuda)

    evaluator = Evaluator(
            "../tables/tagged/",
            "../tables/db/",
            "../stanford-corenlp-full-2018-10-05/"
    )

    if args.test:
        test_exs = em_process(load_dataset(args.test_file))
        test_dataset = dataset.WikiTableDataset(test_exs, vocab)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                               sampler=test_sampler, num_workers=args.data_workers,
                                               collate_fn=dataset.batchify, pin_memory=args.cuda)

        if not args.cuda:
            model = torch.load(args.load_model, map_location={'cuda:0': 'cpu'})
            model.device = device
        else:
            model = torch.load(args.load_model)
        model.to(device)
        f1, pred = evaluate(test_loader, model, evaluator, args.gold_decode)

        with open(args.pred_file, "w") as f:
            json.dump(pred, f, indent=2)

        exit()

    elif args.resume:
        if not args.cuda:
            model = torch.load(args.load_model, map_location={'cuda:0': 'cpu'})
            model.device = device
        else:
            model = torch.load(args.load_model)
        model.to(device)

    else:
        model = TableParser(args, vocab, device)
        model.load_embeddings(args.embed_file)
        model.to(device)

    start_epoch = 0

    params = []
    params_bert = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bert_model' in name:
                params_bert.append(param)
            else:
                params.append(param)
    optimizer = [torch.optim.Adamax(params, lr=1e-3)]
    if len(params_bert):
        optimizer.append(torch.optim.Adamax(params_bert, lr=1e-5))

    logger.info('start training:')
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()
    max_f1 = -np.inf

    ### model training
    for epoch in range(start_epoch, args.num_epochs):
        print_loss_total = 0

        logger.info('start epoch:%d' % epoch)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler, num_workers=args.data_workers,
                                               collate_fn=dataset.batchify, pin_memory=args.cuda)
        model.train()
        model.epoch = epoch

        for idx, batch in enumerate(tqdm(train_loader)):
            loss = model(batch)
            loss = sum(loss)/ batch['word'].size(0)

            for opt in optimizer:
                opt.zero_grad()
            loss.backward()
            for opt in optimizer:
                opt.step()

            clip_grad_norm_(model.parameters(), 5)
            print_loss_total += loss.data.cpu().numpy()
            epoch_loss_total += loss.data.cpu().numpy()
            checkpoint = int(len(train_exs) / (args.batch_size * 2) - 1)

            if idx % checkpoint == 0 and idx != 0:
                f1, pred = evaluate(dev_loader, model, evaluator, args.gold_decode)
                model.train()
                print_loss_avg = print_loss_total / checkpoint
                logger.info('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
                print_loss_total = 0
                if f1 > max_f1:
                    max_f1 = f1
                    with open(args.pred_file, "w") as f:
                        json.dump(pred, f, indent=2)
                    torch.save(model, args.save_model)
