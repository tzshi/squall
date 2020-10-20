# -*- coding: utf-8 -*-

import json
import csv
import re
import numpy as np
from utils import parse_number
import torch
from collections import Counter

from torch.utils.data import Dataset




def _exact_match(col, nl, matched, pos):
    if pos + len(col) > len(nl):
        return False
    if any(matched[pos:pos + len(col)]):
        return False

    if " ".join(col) == " ".join(nl[pos:pos + len(col)]):
        return True

def exact_match(col, nl, matched):
    for i in range(len(nl)):
        if _exact_match(col, nl, matched, i):
            return i
    return -1

def em_process(data):
    for instance in data:
        old_cin = instance['columns_innl']
        old_nic = instance['nl_incolumns']
        nl = instance['nl']

        match_order = [(len(c[1]), i, c[1]) for i, c in enumerate(instance['columns'])]
        nic = [False for i in range(len(instance['nl']))]
        cin = [False for i in range(len(instance['columns']))]

        for _, i, c in sorted(match_order, reverse=True):
            pos = exact_match(c, nl, nic)
            if pos >= 0:
                cin[i] = True
                for j in range(pos, pos+len(c)):
                    nic[j] = True
        left_nl_tokens = set([nl[i] for i in range(len(nl)) if nic[i] is False])
        left_col_tokens = set()
        for i, c in enumerate(instance['columns']):
            left_col_tokens.update(c[1])
        for i, c in enumerate(instance['columns']):
            for x in c[1]:
                if x in left_nl_tokens:
                    cin[i] = True
        for i, x in enumerate(nl):
            if x in left_col_tokens:
                nic[i] = True

        instance['columns_innl'] = cin
        instance['nl_incolumns'] = nic

    return data


def load_dataset(filename):
    with open(filename) as f:
        data = json.load(f)


    
    for instance in data:

        has_number = False
        numbers = []
        for x in instance["nl"]:
            numbers.append(parse_number(x))
            if numbers[-1] is not None:
                has_number = True
        instance["numbers"] = numbers
        instance["has_number"] = has_number


    return data


def load_table(filename):
    with open(filename, "r") as f:
        csv_reader = csv.reader(f)
        table = [row for row in csv_reader]
    return table


def vectorize(instance, vocab):
    word_idx = list()
    word_char_idx = list()
    pos_idx = list()
    ner_idx = list()
    num_idx = list()
    col_type_idx = list()
    nl_in_col = list()
    nl_in_cell = list()
    col_idx = list()
    col_in_nl = list()
    col_char_idx = list()

    words = instance["nl"]
    columns = instance["columns"]
    pos = instance["nl_pos"]
    ner = instance["nl_ner"]
    for w in words:
        word_idx.append(int(vocab['word'].get(w, 1)))
        word_char_idx.append([int(vocab['char'].get(c, 1)) for c in w])
    for p in pos:
        pos_idx.append(int(vocab['pos'].get(p, 1)))
    for e in ner:
        ner_idx.append(int(vocab['ner'].get(e, 1)))
    for c in columns:
        col_type_idx.append([int(vocab['col_type'].get(re.sub(r'<\w+,\s*', '<', c[3]), 1))] * len(c[1]))
    for n in instance["numbers"]:
        num_idx.append(vocab['qtype']["" if n is None else "num"])
    for ico in instance["nl_incolumns"]:
        nl_in_col.append(vocab['qtype']["col" if ico else ""])
    for ice in instance["nl_incells"]:
        nl_in_cell.append(vocab['qtype']["cell" if ice else ""])

    for col, ic in zip(columns, instance["columns_innl"]):
        c_idx = list()
        char_idx = list()
        i_idx = list()
        for w in col[1]:
            c_idx.append(int(vocab['word'].get(w, 1)))
            i_idx.append(vocab['qtype']["nl" if ic else ""])
        for c in col[0]:
            char_idx.append(int(vocab['char'].get(c, 1)))
        char_idx.append(int(vocab['char']['*SEP*']))
        for c in col[4]:
            char_idx.append(int(vocab['char'].get(c, 1)))

        col_idx.append(c_idx)
        col_char_idx.append(char_idx)
        col_in_nl.append(i_idx)
    return {
        'word_idx': word_idx,
        'word_char_idx': word_char_idx,
        'pos_idx': pos_idx,
        'ner_idx': ner_idx,
        'num_idx': num_idx,
        'col_type_idx': col_type_idx,
        'nl_in_col': nl_in_col,
        'nl_in_cell': nl_in_cell,
        'col_idx': col_idx,
        'col_char_idx': col_char_idx,
        'col_in_nl': col_in_nl,
        'sql': instance.get("sql", []) + [["Keyword", "<EOS>", []]],
        'instance': instance,
    }



class WikiTableDataset(Dataset):
    def __init__(self, examples, vocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.vocab)



def batchify(batch):
    word_len = list()
    col_len, col_word_len, col_char_len, word_char_len = list(), list(), list(), list()

    sqls = list()
    instances = list()

    for ex in batch:
        sqls.append(ex['sql'])
        instances.append(ex['instance'])
        word_idx = ex['word_idx']
        word_len.append(len(word_idx))
        col_idx = ex['col_idx']
        col_len.append(len(col_idx))
        for w in col_idx:
            col_word_len.append(len(w))
        for c in ex['col_char_idx']:
            col_char_len.append(len(c))
        for w in ex['word_char_idx']:
            word_char_len.append(len(w))

    word = torch.LongTensor(len(word_len), max(word_len)).zero_()
    word_char = torch.LongTensor(len(word_len), max(word_len), max(word_char_len)).zero_()
    word_char_mask = torch.ByteTensor(len(word_len), max(word_len), max(word_char_len)).fill_(True)
    pos = torch.LongTensor(len(word_len), max(word_len)).zero_()
    ner = torch.LongTensor(len(word_len), max(word_len)).zero_()
    num = torch.LongTensor(len(word_len), max(word_len)).zero_()
    ico = torch.LongTensor(len(word_len), max(word_len)).zero_()
    ice = torch.LongTensor(len(word_len), max(word_len)).zero_()
    col_char = torch.LongTensor(len(word_len), max(col_len), max(col_char_len)).zero_()
    col = torch.LongTensor(len(word_len), max(col_len), max(col_word_len)).zero_()
    inl = torch.LongTensor(len(word_len), max(col_len), max(col_word_len)).zero_()
    col_type = torch.LongTensor(len(word_len), max(col_len), max(col_word_len)).zero_()
    word_mask = torch.ByteTensor(len(word_len), max(word_len)).fill_(True)
    col_char_mask = torch.ByteTensor(len(word_len), max(col_len), max(col_char_len)).fill_(True)
    col_word_mask = torch.ByteTensor(len(word_len), max(col_len), max(col_word_len)).fill_(True)
    col_mask = torch.ByteTensor(len(word_len), max(col_len)).fill_(True)

    for i in range(len(word_len)):
        word_idx = batch[i]['word_idx']
        a1 = torch.LongTensor(word_idx)
        word[i, :len(word_idx)].copy_(a1)
        word_mask[i, :len(word_idx)].fill_(False)

        for k, w_char in enumerate(batch[i]['word_char_idx']):
            a_col = torch.LongTensor(w_char)
            word_char[i, k, :len(w_char)].copy_(a_col)
            word_char_mask[i, k, :len(w_char)].fill_(False)

        pos_idx = batch[i]['pos_idx']
        a1 = torch.LongTensor(pos_idx)
        pos[i, :len(pos_idx)].copy_(a1)

        ner_idx = batch[i]['ner_idx']
        a1 = torch.LongTensor(ner_idx)
        ner[i, :len(ner_idx)].copy_(a1)

        num_idx = batch[i]['num_idx']
        a1 = torch.LongTensor(num_idx)
        num[i, :len(num_idx)].copy_(a1)

        nl_in_col = batch[i]['nl_in_col']
        a1 = torch.LongTensor(nl_in_col)
        ico[i, :len(nl_in_col)].copy_(a1)

        nl_in_cell = batch[i]['nl_in_cell']
        a1 = torch.LongTensor(nl_in_cell)
        ice[i, :len(nl_in_cell)].copy_(a1)

        col_idx = batch[i]['col_idx']
        col_mask[i, :len(col_idx)].fill_(False)
        col_in_nl = batch[i]['col_in_nl']
        col_type_idx = batch[i]['col_type_idx']
        for k, c_word in enumerate(col_idx):
            a_col = torch.LongTensor(c_word)
            col[i, k, :len(c_word)].copy_(a_col)
            col_word_mask[i, k, :len(c_word)].fill_(False)

        for k, c_char in enumerate(batch[i]['col_char_idx']):
            a_col = torch.LongTensor(c_char)
            col_char[i, k, :len(c_char)].copy_(a_col)
            col_char_mask[i, k, :len(c_char)].fill_(False)

        for j, nl in enumerate(col_in_nl):
            a_inl = torch.LongTensor(nl)
            inl[i, j, :len(a_inl)].copy_(a_inl)

        for j, nl in enumerate(col_type_idx):
            a_inl = torch.LongTensor(nl)
            col_type[i, j, :len(a_inl)].copy_(a_inl)
    
    return {
        'word': word,
        'word_char': word_char,
        'word_char_mask': word_char_mask,
        'pos': pos,
        'ner': ner,
        'num': num,
        'ico': ico,
        'ice': ice,
        'col_char': col_char,
        'col': col,
        'inl': inl,
        'col_type': col_type,
        'word_mask': word_mask,
        'col_char_mask': col_char_mask,
        'col_word_mask': col_word_mask,
        'col_mask': col_mask,
        'sqls': sqls,
        'instances': instances,
    }
