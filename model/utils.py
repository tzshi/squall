# -*- coding: utf-8 -*-

import re
import numpy as np
from fuzzywuzzy import fuzz

NUM_MAPPING = {
    'half': 0.5,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'twenty': 20,
    'thirty': 30,
    'once': 1,
    'twice': 2,
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 10,
    'hundred': 100,
    'thousand': 1000,
    'million': 1000000,
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12,
}


def get_cells(table):
    ret = set()
    for content in table["contents"][2:]:
        for col in content:
            if col["type"] == "TEXT":
                for x in col["data"]:
                    ret.add((col["col"], str(x)))
            elif col["type"] == "LIST TEXT":
                for lst in col["data"]:
                    for x in lst:
                        ret.add((col["col"], str(x)))

    return ret



def best_match(candidates, query, col=None):
    # return max(candidates, key=lambda x: fuzz.partial_ratio(x, query))
    return max(candidates, key=lambda x: (fuzz.ratio(x[1], query), col==x[0]))


def parse_number(s):
    if s in NUM_MAPPING:
        return NUM_MAPPING[s]

    s = s.replace(',', '')
    # https://stackoverflow.com/questions/4289331/python-extract-numbers-from-a-string
    ret = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
    if len(ret) > 0:
        return ret[0]

    return None

def attention_matrix_col(instance):
    w_att_c = np.zeros((len(instance["nl"]), len(instance["columns"])))
    c_attn_w = np.zeros((len(instance["columns"]), len(instance["nl"])))
    for ystep, (ttype, value, span) in enumerate(instance['sql']):
        if ttype == 'Column':
            s = value.split("_")
            col = int(s[0][1:]) - 1
            for xs, ys in instance['align']:
                if ystep in ys:
                    for x in xs:
                        w_att_c[x, col] = 1.
                        c_attn_w[col, x] = 1.

    for i in range(w_att_c.shape[0]):
        if sum(w_att_c[i]):
            w_att_c[i] /= sum(w_att_c[i])


    for i in range(c_attn_w.shape[0]):
        if sum(c_attn_w[i]):
            c_attn_w[i] /= sum(c_attn_w[i])

    return w_att_c, c_attn_w


def attention_matrix(instance):
    att = np.zeros((len(instance["sql"]) + 1, len(instance["nl"])))

    for xs, ys in instance["align"]:
        for x in xs:
            for y in ys:
                att[y, x] = 1.

    for i in range(att.shape[0]):
        if sum(att[i]):
            att[i] /= sum(att[i])

    return att

