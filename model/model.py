import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import json
import copy
import math
import numpy as np
from utils import parse_number, get_cells, best_match, attention_matrix, attention_matrix_col
from transformers import BertTokenizer, BertModel

if torch.cuda.is_available():
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(async=True)
else:
    from torch import from_numpy


def l2_loss(x, y):
    ret = 0.
    for i in range(len(y)):
        diff = x[i] - y[i]
        ret = ret + diff * diff
    return [0.5 * ret]

def ce_loss(x, y):
    ret = 0.
    for i in range(len(y)):
        if y[i] > 0.:
            ret = ret + y[i] * torch.log(x[i] + 1e-8)
    return [-.2 * ret]

def mul_loss(x, y):
    ret = 0.
    for i in range(len(y)):
        if y[i] > 0.:
            ret = ret + x[i]
    if ret == 0:
        return []
    ret = ret + 1e-8
    return [-1. * torch.log(ret)]


class TableParser(nn.Module):


    def __init__(self, args, vocab, device):
        super(TableParser, self).__init__()
        ## Model parameters

        self._batch_size = 8
        self.vocab = vocab
        self.args = args

        self._cdim = 32
        self._wdim = 100
        self._idim = 256

        self._qtdim = 8

        self._bilstm_dim = 128

        self._lstm_layer = 2
        self._mlp_layer = 1
        self._mlp_dim = 128

        self._lstm_dropout = 0.2
        self._mlp_dropout = 0.2

        self._coverage = False

        self.device = device


        self._char_lookup = nn.Embedding(num_embeddings=len(self.vocab['char']), embedding_dim=self._cdim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._word_lookup = nn.Embedding(num_embeddings=len(self.vocab['word']), embedding_dim=self._wdim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._keyword_lookup = nn.Embedding(num_embeddings=len(self.vocab['keyword']), embedding_dim=self._idim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._pos_lookup = nn.Embedding(num_embeddings=len(self.vocab['pos']), embedding_dim=self._qtdim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._ner_lookup = nn.Embedding(num_embeddings=len(self.vocab['ner']), embedding_dim=self._qtdim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._qtype_lookup = nn.Embedding(num_embeddings=len(self.vocab['qtype']), embedding_dim=self._qtdim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._ctype_lookup = nn.Embedding(num_embeddings=len(self.vocab['col_type']), embedding_dim=self._qtdim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._decoder_lookup = nn.Embedding(num_embeddings=1, embedding_dim=self._idim, max_norm=None, scale_grad_by_freq=False, sparse=False)

        q_input_dim = self._wdim + self._qtdim * 5 + self._bilstm_dim * 2
        if self.args.bert:
            self._bert_model = BertModel.from_pretrained("bert-base-uncased")
            self._bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
            self._bert_w_project = nn.Linear(self._bert_model.pooler.dense.in_features, self._bilstm_dim * 2, bias=False)
            self._bert_c_project = nn.Linear(self._bert_model.pooler.dense.in_features, self._bilstm_dim * 2, bias=False)

            self.c_attn_w = MatchAttn(self._bilstm_dim * 2, identity=True)
            self.w_attn_c = MatchAttn(self._bilstm_dim * 2, identity=True)

            self._q_bilstm_2 = Encoder_rnn(self.args, self._bilstm_dim * 4, self._bilstm_dim)
            self._c_bilstm_2 = Encoder_rnn(self.args, self._bilstm_dim * 4, self._bilstm_dim)
        else:

            self._q_charlstm = Encoder_rnn(self.args, self._cdim, self._bilstm_dim)
            self._c_charlstm = Encoder_rnn(self.args, self._cdim, self._bilstm_dim)

            self._q_bilstm = Encoder_rnn(self.args, q_input_dim, self._bilstm_dim)
            self._c_bilstm = Encoder_rnn(self.args, self._wdim + self._qtdim * 2 + self._bilstm_dim * 2, self._bilstm_dim)



            self.c_self_attn = MatchAttn(self._bilstm_dim * 2, identity=True)
            self.c_attn_w = MatchAttn(self._bilstm_dim * 2, identity=True)
            self.w_attn_c = MatchAttn(self._bilstm_dim * 2, identity=True)


            self._q_bilstm_2 = Encoder_rnn(self.args, self._bilstm_dim * 4, self._bilstm_dim)
            self._c_bilstm_2 = Encoder_rnn(self.args, self._bilstm_dim * 4, self._bilstm_dim)

        self.decode_c_attn = MatchAttn(self._bilstm_dim * 2, identity=True)
        self.decode_w_attn = MultiHeadedAttention(1, self._bilstm_dim * 2, dropout=0., coverage=self._coverage)

        self._h_lstm = nn.LSTM(self._idim, self._bilstm_dim * 2, 2, dropout=self._lstm_dropout)

        self._type_final = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 6, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, len(self.vocab['type']))
        )
        self._keyword_final = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 6, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, len(self.vocab['keyword']))
        )
        self._columnt_final = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 6, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, len(self.vocab['columnt']))
        )

        if self.args.aux_col:
            self._aux_col = MatchAttn(self._bilstm_dim * 2, self._bilstm_dim * 2)

        self._column_biaffine = MatchAttn(self._bilstm_dim * 6, self._bilstm_dim * 2)
        self._valbeg_biaffine = MatchAttn(self._bilstm_dim * 6, self._bilstm_dim * 2)
        self._valend_biaffine = MatchAttn(self._bilstm_dim * 6, self._bilstm_dim * 2)

        self._column2i = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 2, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, self._idim)
        )


        #MultiLayerPerceptron([self._bilstm_dim] + [self._mlp_dim] * self._mlp_layer + [self._idim], dy.rectify, self._model)
        self._val2i = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 4, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, self._idim)
        )

        self.epoch = 0


    def load_embeddings(self, filename):
        with open(filename, 'r') as f:
            emb = json.load(f)
        count = 0
        embedding = self._word_lookup.weight.data
        for w in emb:
            if w in self.vocab['word'] and len(emb[w]) == self._wdim:
                count += 1
                w_t = torch.Tensor([float(i) for i in emb[w]])
                embedding[self.vocab['word'][w]].copy_(w_t)

        print("Init", count, "embeddings")

    def _bert_features(self, batch, isTrain=False):
        instances = batch['instances']

        word_seq_max_len = max([len(ins["nl"]) for ins in instances])
        col_seq_max_len = max([len(ins["columns"]) for ins in instances])

        all_input_ids = np.zeros((len(instances), 2048), dtype=int)
        all_input_type_ids = np.zeros((len(instances), 2048), dtype=int)
        all_input_mask = np.zeros((len(instances), 2048), dtype=int)
        all_word_end_mask = np.zeros((len(instances), 2048), dtype=int)
        all_col_end_mask = np.zeros((len(instances), 2048), dtype=int)

        subword_max_len = 0

        for snum, ins in enumerate(instances):
            question_words = ins["nl"]

            tokens = []
            token_types = []
            word_end_mask = []
            col_end_mask = []

            tokens.append("[CLS]")
            token_types.append(0)
            word_end_mask.append(0)
            col_end_mask.append(0)

            for word in question_words:
                word_tokens = self._bert_tokenizer.tokenize(word)
                if len(word_tokens) == 0:
                    word_tokens = ['.']
                for _ in range(len(word_tokens)):
                    word_end_mask.append(0)
                    col_end_mask.append(0)
                    token_types.append(0)
                word_end_mask[-1] = 1
                tokens.extend(word_tokens)

            tokens.append("[SEP]")
            word_end_mask.append(0)
            col_end_mask.append(0)
            token_types.append(0)

            for col in ins["columns"]:
                col_tokens = self._bert_tokenizer.tokenize(col[0])
                if len(col_tokens) == 0:
                    col_tokens = ['.']
                for _ in range(len(col_tokens)):
                    word_end_mask.append(0)
                    col_end_mask.append(0)
                    token_types.append(1)
                col_end_mask[-1] = 1
                tokens.extend(col_tokens)

                tokens.append("[SEP]")
                word_end_mask.append(0)
                col_end_mask.append(0)
                token_types.append(1)

            # pad to sequence length for every sentence
            for i in range(word_seq_max_len - len(ins["nl"])):
                word_end_mask.append(1)
            for i in range(col_seq_max_len - len(ins["columns"])):
                col_end_mask.append(1)

            input_ids = self._bert_tokenizer.convert_tokens_to_ids(tokens)
            input_type_ids = token_types
            input_mask = [1] * len(input_ids)

            subword_max_len = max(max(subword_max_len, len(word_end_mask) + 1), len(col_end_mask) + 1)

            all_input_ids[snum, :len(input_ids)] = input_ids
            all_input_type_ids[snum, :len(input_type_ids)] = input_type_ids
            all_input_mask[snum, :len(input_mask)] = input_mask
            all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask
            all_col_end_mask[snum, :len(col_end_mask)] = col_end_mask

        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len]))
        all_input_type_ids = from_numpy(np.ascontiguousarray(all_input_type_ids[:, :subword_max_len]))
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len]))
        all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len]))
        all_col_end_mask = from_numpy(np.ascontiguousarray(all_col_end_mask[:, :subword_max_len]))
        features, _ = self._bert_model(all_input_ids, token_type_ids=all_input_type_ids, attention_mask=all_input_mask)
        del _


        bert_word_features = features.masked_select(all_word_end_mask.to(torch.bool).unsqueeze(-1)).reshape(len(instances), word_seq_max_len, features.shape[-1])
        bert_col_features = features.masked_select(all_col_end_mask.to(torch.bool).unsqueeze(-1)).reshape(len(instances), col_seq_max_len, features.shape[-1])


        # BERT encoding for question and table
        wvecs = self._bert_w_project(bert_word_features)
        cvecs = self._bert_c_project(bert_col_features)

        w_mask = batch['word_mask'].to(self.device)
        c_mask = batch['col_mask'].to(self.device)
        # Attention
        wcontext, alpha_w = self.w_attn_c(wvecs, cvecs, c_mask)
        ccontext, alpha_c = self.c_attn_w(cvecs, wvecs, w_mask)

        # Supervised Attn
        loss = list()
        if isTrain and self.args.enc_loss:

            for i, instance in enumerate(batch['instances']):
                w_att_c_matrix, c_att_w_matrix = attention_matrix_col(instance)
                w_sp0, w_sp1 = w_att_c_matrix.shape
                for stp in range(w_sp0):
                    ll = l2_loss(alpha_w[i][stp, :w_sp1], w_att_c_matrix[stp])
                    loss.extend(ll)
                c_sp0, c_sp1 = c_att_w_matrix.shape
                for stp in range(c_sp0):
                    ll = l2_loss(alpha_c[i][stp, :c_sp1], c_att_w_matrix[stp])
                    loss.extend(ll)

        wvecs_pre = wvecs
        cvecs_pre = cvecs

        wvecs = torch.cat((wvecs, wcontext), 2)
        cvecs = torch.cat((cvecs, ccontext), 2)

        w_len = batch['word_mask'].data.eq(0).long().sum(1).numpy().tolist()
        wvecs = self._q_bilstm_2(wvecs, w_len)
        c_len = batch['col_mask'].data.eq(0).long().sum(1).numpy().tolist()
        cvecs = self._c_bilstm_2(cvecs, c_len)

        return wvecs_pre, cvecs_pre, wvecs, cvecs, loss

    ###LSTM Encoder part
    def _lstm_features(self, batch, isTrain=False):
        words = batch['word'].to(self.device)
        word_emb = self._word_lookup(words)

        word_char = batch['word_char'].to(self.device)
        word_char_emb = self._char_lookup(word_char)

        pos = batch['pos'].to(self.device)
        pos_emb = self._pos_lookup(pos)

        ner = batch['ner'].to(self.device)
        ner_emb = self._ner_lookup(ner)

        is_num = batch['num'].to(self.device)
        num_emb = self._qtype_lookup(is_num)

        ico = batch['ico'].to(self.device)
        ico_emb = self._qtype_lookup(ico)

        ice = batch['ice'].to(self.device)
        ice_emb = self._qtype_lookup(ice)

        word_len = batch['word_mask'].data.eq(0).long().sum(1).numpy().tolist()

        charvecs = word_char_emb
        batch_size, col_len, col_word_len = charvecs.size(0), charvecs.size(1), charvecs.size(2)
        charvecs = charvecs.view(batch_size * col_len, col_word_len, -1)
        cvecs_len = batch['word_char_mask'].view(batch_size * col_len, col_word_len).data.eq(0).long().sum(1).numpy().tolist()
        cvecs_idx = [i for i in range(len(cvecs_len)) if cvecs_len[i] > 0]
        cvecs_len = [i for i in cvecs_len if i > 0]
        charvecs = charvecs[cvecs_idx]
        char_rep = self._q_charlstm(charvecs, cvecs_len, all_hiddens=False)
        charvecs = torch.FloatTensor(batch_size * col_len, 2 * self._bilstm_dim).zero_().to(self.device)
        charvecs[cvecs_idx] = char_rep
        word_charvecs = charvecs.view(batch_size, col_len, -1)

        ## The input is the combination of multiple embeddings
        wvecs = torch.cat((word_emb, pos_emb, ner_emb, num_emb, ico_emb, ice_emb, word_charvecs), 2)
        ## question input LSTM
        wvecs = self._q_bilstm(wvecs, word_len)

        col = batch['col'].to(self.device)
        col_emb = self._word_lookup(col)
        col_char = batch['col_char'].to(self.device)
        col_char_emb = self._char_lookup(col_char)

        inl = batch['inl'].to(self.device)
        inl_emb = self._qtype_lookup(inl)

        col_type = batch['col_type'].to(self.device)
        col_type_emb = self._ctype_lookup(col_type)


        charvecs = col_char_emb
        batch_size, col_len, col_word_len = charvecs.size(0), charvecs.size(1), charvecs.size(2)
        charvecs = charvecs.view(batch_size * col_len, col_word_len, -1)
        cvecs_len = batch['col_char_mask'].view(batch_size * col_len, col_word_len).data.eq(0).long().sum(1).numpy().tolist()
        cvecs_idx = [i for i in range(len(cvecs_len)) if cvecs_len[i] > 0]
        cvecs_len = [i for i in cvecs_len if i > 0]
        charvecs = charvecs[cvecs_idx]
        char_rep = self._c_charlstm(charvecs, cvecs_len, all_hiddens=False)
        charvecs = torch.FloatTensor(batch_size * col_len, 2 * self._bilstm_dim).zero_().to(self.device)
        charvecs[cvecs_idx] = char_rep
        charvecs = charvecs.view(batch_size, col_len, -1).unsqueeze(2).expand(-1, -1, col_emb.shape[2], -1)

        cvecs = torch.cat((col_emb, inl_emb, col_type_emb, charvecs), 3)
        batch_size, col_len, col_word_len = cvecs.size(0), cvecs.size(1), cvecs.size(2)
        cvecs = cvecs.view(batch_size * col_len, col_word_len, -1)
        cvecs_len = batch['col_word_mask'].view(batch_size * col_len, col_word_len).data.eq(0).long().sum(1).numpy().tolist()
        cvecs_idx = [i for i in range(len(cvecs_len)) if cvecs_len[i] > 0]
        cvecs_len = [i for i in cvecs_len if i > 0]
        cvecs = cvecs[cvecs_idx]
        ### Table input LSTM
        c_rep = self._c_bilstm(cvecs, cvecs_len, all_hiddens=False)

        cvecs = torch.FloatTensor(batch_size * col_len, 2 * self._bilstm_dim).zero_().to(self.device)
        cvecs[cvecs_idx] = c_rep
        cvecs = cvecs.view(batch_size, col_len, -1)

        w_mask = batch['word_mask'].to(self.device)
        c_mask = batch['col_mask'].to(self.device)


        loss = list()
        ##Table input self attn
        c_attn, _ = self.c_self_attn(cvecs, cvecs, c_mask)
        cvecs = cvecs + c_attn

        ## question2table and table2question attentions

        wcontext, alpha_w = self.w_attn_c(wvecs, cvecs, c_mask)
        c_Score = self.c_attn_w(cvecs, wvecs, w_mask, is_score=True)
        ccontext, alpha_c = self.c_attn_w(cvecs, wvecs, w_mask)

        ### Encoder supervised attention
        if isTrain and self.args.enc_loss:

            for i, instance in enumerate(batch['instances']):
                w_att_c_matrix, c_att_w_matrix = attention_matrix_col(instance)
                w_sp0, w_sp1 = w_att_c_matrix.shape
                for stp in range(w_sp0):
                    ll = l2_loss(alpha_w[i][stp, :w_sp1], w_att_c_matrix[stp])
                    loss.extend(ll)
                c_sp0, c_sp1 = c_att_w_matrix.shape
                for stp in range(c_sp0):
                    ll = l2_loss(alpha_c[i][stp, :c_sp1], c_att_w_matrix[stp])
                    loss.extend(ll)

        wvecs_pre = wvecs
        cvecs_pre = cvecs

        wvecs = torch.cat((wvecs, wcontext), 2)
        cvecs = torch.cat((cvecs, ccontext), 2)

        ## Another LSTM layer
        wvecs = self._q_bilstm_2(wvecs, word_len)
        c_len = batch['col_mask'].data.eq(0).long().sum(1).numpy().tolist()
        cvecs = self._c_bilstm_2(cvecs, c_len)

        return wvecs_pre, cvecs_pre, wvecs, cvecs, loss


    def forward(self, batch, isTrain=True, gold_decode=False):
        if self.args.bert:
            wvecs_pre, cvecs_pre, wvecs, cvecs, loss = self._bert_features(batch, isTrain=isTrain)
        else:
            wvecs_pre, cvecs_pre, wvecs, cvecs, loss = self._lstm_features(batch, isTrain=isTrain)
        sqls = batch['sqls']
        instances = batch['instances']


        zeros = torch.zeros(1, self._bilstm_dim * 2, device=self.device)
        criterion = nn.NLLLoss(ignore_index=-1, reduction="sum")
        if isTrain:
            for i in range(len(batch['sqls'])):
                # Column prediction component
                if self.args.aux_col:

                    potential = self._aux_col(wvecs[i].unsqueeze(0), cvecs[i].unsqueeze(0), batch['col_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0)

                    label = [int(x[1].split("_")[0][1:]) - 1 if x[0] =='Column' else -1 for x in instances[i]["nl_ralign"]]
                    for _ in range(len(instances[i]["nl_ralign"]), potential.shape[0]):
                        label.append(-1)
                    loss.append(0.2 * criterion(F.log_softmax(potential, dim=1), torch.tensor(label, device=self.device)))

                att = attention_matrix(instances[i])
                decoder_input = self._decoder_lookup(torch.tensor([0], device=self.device))
                hidden = None

                for ystep, (ttype, value, span) in enumerate(sqls[i]):


                    hvec, c_hidden = self._h_lstm(decoder_input.unsqueeze(1), hidden)
                    hidden = c_hidden


                    val = wvecs[i].unsqueeze(0)
                    if self.args.gold_attn:
                        w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i], gold_attn=att[ystep])
                    else:
                        w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i])
                    c_context, c_score = self.decode_c_attn(hvec, cvecs[i].unsqueeze(0), batch['col_mask'].to(self.device)[i].unsqueeze(0))


                    # Decoder supervised attn
                    if self.args.dec_loss and not self.args.gold_attn:
                        w_score = self.decode_w_attn.attn.squeeze(0).squeeze(1)[0]
                        #ll = mul_loss(w_score, att[ystep + 1])
                        ll = l2_loss(w_score, att[ystep])
                        #ll = ce_loss(w_score, att[ystep])
                        loss.extend(ll)

                    hvec = torch.cat((hvec, w_context, c_context), -1)

                    # decide current prediction token type
                    potential = self._type_final(hvec).squeeze(0)

                    loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([self.vocab['type'][ttype]], device=self.device)))

                    # Keyword prediction
                    if ttype == "Keyword":
                        potential = self._keyword_final(hvec).squeeze(0)
                        loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([self.vocab['keyword'][value]], device=self.device)))
                        k_value = torch.tensor([self.vocab['keyword'][value]], device=self.device)

                        ivec = self._keyword_lookup(k_value)

                    # Column prediction
                    elif ttype == "Column":
                        s = value.split("_")
                        col = int(s[0][1:]) - 1
                        if len(s) > 1:
                            columnt = '_'.join(s[1:])
                        else:
                            columnt = ""
                        columnt_candidates = instances[i]['columns'][col][2] + [""]

                        if len(columnt_candidates) > 1:
                            potential = self._columnt_final(hvec).squeeze(0)
                            mask = np.zeros((len(self.vocab['columnt']), ))
                            cand_set = set()
                            for t in columnt_candidates:
                                if t in self.vocab['columnt']:
                                    cand_set.add(self.vocab['columnt'][t])
                            for j in range(len(self.vocab['columnt'])):
                                if j not in cand_set:
                                    mask[j] = -1000.
                            potential = potential + torch.tensor(mask, device=self.device).float()
                            loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([self.vocab['columnt'][columnt]], device=self.device)))


                        potential = self._column_biaffine(hvec, cvecs[i].unsqueeze(0), batch['col_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0)
                        loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([col], device=self.device)))

                        ivec = self._column2i(cvecs[i][col].unsqueeze(0))

                    # Literal Prediction
                    else:
                        potential = self._valbeg_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0)
                        loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([span[0]], device=self.device)))

                        if len(span) > 1:
                            potential = self._valend_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0)
                            loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([span[1]], device=self.device)))

                            ivec = self._val2i(torch.cat((wvecs[i][span[0]].unsqueeze(0), wvecs[i][span[1]].unsqueeze(0)), 1))
                        else:
                            ivec = self._val2i(torch.cat((wvecs[i][span[0]].unsqueeze(0), wvecs[i][span[0]].unsqueeze(0)), 1))

                    decoder_input = ivec

            return loss
        else:
            logprobs = []
            pred_data = list()
            for i in range(len(batch['sqls'])):
                query = []
                types = []
                ii = 0
                json_file = "../tables/json/{}.json".format(instances[i]["tbl"])
                with open(json_file, "r") as f:
                    table = json.load(f)

                cells = get_cells(table)
                hidden = None
                decoder_input = self._decoder_lookup(torch.tensor([0], device=self.device))
                if self.args.gold_attn:
                    att = attention_matrix(instances[i])

                if gold_decode:
                    gold_sql = batch['sqls'][i]


                while True:
                    if gold_decode:
                        gold_ttype, gold_value, gold_span = gold_sql[ii]
                    logprob = 0.
                    ii += 1
                    if ii> 100:
                        break

                    hvec, c_hidden = self._h_lstm(decoder_input.unsqueeze(1), hidden)
                    hidden = c_hidden


                    val = wvecs[i].unsqueeze(0)
                    if self.args.gold_attn:
                        if ii - 1 < att.shape[0]:
                            w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i], gold_attn=att[ii - 1])
                        else:
                            w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i], gold_attn=att[-1])
                    else:
                        w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i])
                    c_context, c_score = self.decode_c_attn(hvec, cvecs[i].unsqueeze(0), batch['col_mask'].to(self.device)[i].unsqueeze(0))
                    w_score = self.decode_w_attn.attn.squeeze(0).squeeze(1)[0]



                    hvec = torch.cat((hvec, w_context, c_context), -1)

                    potential = F.log_softmax(self._type_final(hvec).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()

                    if not instances[i]["has_number"]:
                        potential[self.vocab['type']["Literal.Number"]] = -np.inf

                    if gold_decode:
                        choice = self.vocab['type'][gold_ttype]
                    else:
                        choice = np.argmax(potential)

                    ttype = list(self.vocab['type'].keys())[choice]
                    types.append(ttype)
                    logprob += potential[choice]

                    if ttype == "Keyword":

                        potential = F.log_softmax(self._keyword_final(hvec).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                        if gold_decode:
                            iv = self.vocab['keyword'][gold_value]
                        else:
                            iv = np.argmax(potential)

                        value = list(self.vocab['keyword'].keys())[iv]

                        logprob += potential[iv]

                        if value == "<EOS>":
                            logprobs.append(logprob)
                            break

                        query.append(value)

                        k_value = torch.tensor([self.vocab['keyword'][value]], device=self.device)
                        ivec = self._keyword_lookup(k_value)

                    elif ttype == "Column":

                        potential = F.log_softmax(self._column_biaffine(hvec, cvecs[i].unsqueeze(0), batch['col_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                        if gold_decode:
                            col = int(gold_value.split("_")[0][1:]) - 1
                        else:
                            col = np.argmax(potential)

                        logprob += potential[col]

                        columnt_candidates = instances[i]['columns'][col][2] + [""]

                        if len(columnt_candidates) > 1:
                            potential = self._columnt_final(hvec).squeeze(0)
                            mask = np.zeros((len(self.vocab['columnt']), ))
                            cand_set = set()
                            for t in columnt_candidates:
                                if t in self.vocab['columnt']:
                                    cand_set.add(self.vocab['columnt'][t])
                            for j in range(len(self.vocab['columnt'])):
                                if j not in cand_set:
                                    mask[j] = -1000.
                            potential = potential + torch.tensor(mask, device=self.device).float()
                            potential = F.log_softmax(potential, dim=1).squeeze(0).data.cpu().numpy()

                            if gold_decode:
                                s = gold_value.split("_")
                                gold_columnt = "_".join(s[1:])
                                if len(s) > 1:
                                    if gold_columnt in self.vocab['columnt']:
                                        choice = self.vocab['columnt'][gold_columnt]
                                        logprob += potential[choice]
                                else:
                                    choice = self.vocab['columnt'][""]
                                    logprob += potential[choice]
                                columnt = gold_columnt
                            else:
                                choice = np.argmax(potential)
                                columnt = list(self.vocab['columnt'].keys())[choice]
                                logprob += potential[choice]
                        else:
                            columnt = ""


                        if columnt != "":
                            query.append("c{}_{}".format(col + 1, columnt))
                        else:
                            query.append("c{}".format(col + 1))

                        ivec = self._column2i(cvecs[i][col]).unsqueeze(0)

                    else:

                        if ttype == "Literal.String":
                            potential = F.log_softmax(self._valbeg_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                            if gold_decode:
                                span_beg = gold_span[0]
                            else:
                                span_beg = np.argmax(potential)
                            logprob += potential[span_beg]

                            potential = F.log_softmax(self._valend_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                            if gold_decode:
                                span_end = gold_span[1]
                            else:
                                span_end = np.argmax(potential[span_beg:]) + span_beg
                            logprob += potential[span_end]

                            if len(query) >= 2 and query[-1] == "=" and types[-3] == "Column":
                                col, literal = best_match(cells, " ".join(instances[i]["nl"][span_beg:span_end+1]), query[-2])
                            else:
                                col, literal = best_match(cells, " ".join(instances[i]["nl"][span_beg:span_end+1]))

                            query.append("{}".format(repr(literal)))


                            # postprocessing, fix the col = val mismatch
                            if len(query) >= 3 and query[-2] == "=" and types[-3] == "Column":
                                query[-3] = col

                            if len(query) >= 4 and query[-2] == "(" and query[-3] == "in" and types[-4] == "Column":
                                query[-4] = col
                        else:
                            potential = F.log_softmax(self._valbeg_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                            for j, n in enumerate(instances[i]["numbers"]):
                                if n is None:
                                    potential[j] = -np.inf

                            if gold_decode:
                                span_beg = gold_span[0]
                            else:
                                span_beg = np.argmax(potential)
                            logprob += potential[span_beg]
                            span_end = span_beg
                            query.append("{}".format(parse_number(instances[i]["nl"][span_beg])))
                        ivec = self._val2i(torch.cat((wvecs[i][span_beg].unsqueeze(0), wvecs[i][span_end].unsqueeze(0)), 1))

                    decoder_input = ivec
                    logprobs.append(logprob)



                _query = query
                types = " ".join(types)
                query = ' '.join(query)

                pred_data.append({
                'table_id': instances[i]["tbl"],
                'result': [
                        {
                        'sql': query,
                        'sql_type': types,
                        'id': instances[i]["nt"],
                        'tgt': " ".join([x[1] for x in instances[i].get("sql", [])]),
                        'nl': ' '.join(instances[i]['nl'])
                        }
                    ]
                })

            return pred_data, logprobs


class Encoder_rnn(nn.Module):
    def __init__(self, args, input_size, hidden_size):
        super(Encoder_rnn, self).__init__()
        self.args = args


        self.rnn = nn.LSTM(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = self.args.num_layers,
                          batch_first = True,
                          dropout = self.args.dropout,
                          bidirectional = True)

    def forward(self, emb, emb_len, all_hiddens=True):

        emb_len = np.array(emb_len)
        sorted_idx= np.argsort(-emb_len)
        emb = emb[sorted_idx]
        emb_len = emb_len[sorted_idx]
        unsorted_idx = np.argsort(sorted_idx)

        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(emb, emb_len, batch_first=True)
        output, hn = self.rnn(packed_emb)

        if all_hiddens:
            unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
            unpacked = unpacked.transpose(0, 1)
            unpacked = unpacked[torch.LongTensor(unsorted_idx)]
            return unpacked
        else:
            ret = hn[0][-2:].transpose(1,0).contiguous()[torch.LongTensor(unsorted_idx)]
            ret = ret.view(ret.shape[0], -1)
            return ret


'''
From DrQA repo https://github.com/facebookresearch/DrQA
Single head attention
'''
class MatchAttn(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size1, input_size2=None, identity=False):
        super(MatchAttn, self).__init__()
        if input_size2 is None:
            input_size2 = input_size1
        hidden_size = min(input_size1, input_size2)
        if not identity:
            self.linear_x = nn.Linear(input_size1, hidden_size)
            self.linear_y = nn.Linear(input_size2, hidden_size)
        else:
            self.linear_x = None
            self.linear_y = None

        self.w = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, y, y_mask, is_score=False, no_diag=False):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear_x:
            x_proj = self.linear_x(x.view(-1, x.size(2))).view(x.shape[0], x.shape[1], -1)
            x_proj = F.relu(x_proj)
            y_proj = self.linear_y(y.view(-1, y.size(2))).view(y.shape[0], y.shape[1], -1)
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        x_proj = self.w(x_proj)

        # Compute scores

        scores = x_proj.bmm(y_proj.transpose(2, 1))
        # Mask padding

        y_mask = y_mask.unsqueeze(1).expand(scores.size())

        scores.data.masked_fill_(y_mask.bool().data, -float('inf'))
        # Normalize with softmax
        if is_score:
            return scores
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))


        # Take weighted average
        matched_seq = alpha.bmm(y)
        #residual_rep = torch.abs(x - matched_seq)

        return matched_seq, alpha


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, gate=None, gold_attn=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask.bool(), -float('inf'))
    p_attn = F.softmax(scores, dim = -1)

    if gate:
        gated = torch.sigmoid(gate(query))
        p_attn = p_attn * gated

    if dropout is not None:
        p_attn = dropout(p_attn)

    if gold_attn is not None:
        length = gold_attn.shape[0]
        first_head = p_attn[:, 0:1, :, :]
        rest = p_attn[:, 1:, :, :]

        new_first = torch.zeros_like(first_head)
        new_first[0][0][0][:length] = torch.Tensor(gold_attn[:length])
        p_attn = torch.cat((new_first, rest), 1)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, gate=False, coverage=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        if gate:
            self.gate = nn.Linear(self.d_k, 1)
        else:
            self.gate = None

        if coverage:
            self.linear_cover = nn.Linear(1, self.d_k)
        else:
            self.linear_cover = None

    def forward(self, query, key, value, mask=None, coverage=None, gold_attn=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            # mask = mask.unsqueeze(1)
            pass
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # coverage
        if coverage is not None:
            key += self.linear_cover(coverage.unsqueeze(-1))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout, gold_attn=gold_attn)
                                 # dropout=self.dropout, gate=self.gate)
        #print(self.attn)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
