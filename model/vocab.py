# -*- coding: utf-8 -*-

from collections import Counter
import re


def build_vocab(dataset, cutoff=1):
    print("Building vocab from", len(dataset), "instances")

    char_counter = Counter()
    word_counter = Counter()
    pos_counter = Counter()
    ner_counter = Counter()
    keyword_counter = Counter()
    columnt_counter = Counter()
    type_counter = Counter()
    col_type_counter = Counter()

    for instance in dataset:
        word_counter.update(instance["nl"])
        for w in instance["nl"]:
            char_counter.update(w)
        pos_counter.update(instance["nl_pos"])
        ner_counter.update(instance["nl_ner"])

        for c in instance["columns"]:
            word_counter.update(c[1])
            char_counter.update(c[0])
            char_counter.update(c[4])
            columnt_counter.update(c[2])
            col_type_counter[re.sub(r'<\w+,\s*', '<', c[3])] += 1

        if "sql" in instance:
            for y in instance["sql"]:
                type_counter[y[0]] += 1
                if y[0] == "Keyword":
                    keyword_counter[y[1]] += 1

    print("Charset contains", len(char_counter), "chars")
    print("Vocab contains", len(word_counter), "words")

    char_counter = Counter({
        w: i for w, i in char_counter.items() if i >= cutoff})

    word_counter = Counter({
        w: i for w, i in word_counter.items() if i >= cutoff})

    col_type_counter = Counter({
        w: i for w, i in col_type_counter.items() if i >= cutoff})

    print("Charset contains", len(char_counter), "words after cutting off")
    print("Vocab contains", len(word_counter), "words after cutting off")
    print("POS:", pos_counter)
    print("NER:", ner_counter)
    print("Keywords:", keyword_counter)
    print("Column SubTypes:", columnt_counter)
    print("Column Types:", col_type_counter)
    print("SQL Token Types:", type_counter)

    char_vocab = list(char_counter.keys())
    word_vocab = list(word_counter.keys())
    pos_vocab = list(pos_counter.keys())
    ner_vocab = list(ner_counter.keys())
    keyword_vocab = list(keyword_counter.keys())
    columnt_vocab = list(columnt_counter.keys())
    col_type_vocab = list(col_type_counter.keys())
    type_vocab = list(type_counter.keys())

    _ichar = ["*PAD*", "*UNK*", "*SEP*"] + char_vocab
    _iword = ["*PAD*", "*UNK*"] + word_vocab
    _ipos = ["*PAD*", "*UNK*"] + pos_vocab
    _iner = ["*PAD*", "*UNK*"] + ner_vocab
    _icol_type = ["*PAD*", "*UNK*"] + col_type_vocab
    _iqtype = ["*PAD*", "", "num", "col", "cell", "nl"]

    _ikeyword = ["<EOS>"] + keyword_vocab
    _icolumnt = [""] + columnt_vocab
    # _ilexicon = ["*UNK*"] + lexicon_vocab

    ret = {
        "char": {w: i for i, w in enumerate(_ichar)},
        "word": {w: i for i, w in enumerate(_iword)},
        "pos": {w: i for i, w in enumerate(_ipos)},
        "ner": {w: i for i, w in enumerate(_iner)},
        "keyword": {w: i for i, w in enumerate(_ikeyword)},
        "columnt": {w: i for i, w in enumerate(_icolumnt)},
        "col_type": {w: i for i, w in enumerate(_icol_type)},
        "type": {w: i for i, w in enumerate(type_vocab)},
        "qtype": {w: i for i, w in enumerate(_iqtype)},
    }

    return ret


