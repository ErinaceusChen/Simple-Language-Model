#! /usr/bin/python
# -*- coding: utf-8 -*-
# @ FileName    : word2num.py
# @ CreatedDate : 2019-04-07 18:56
# @ Author      : Joshua Chan
# @ Company     : PingAn OneConnect

import codecs

raw_data = "data/ptb.test.txt"  # origin train data corpus
vocab = 'data/ptb.vocab'  # corpus vocabulary
output_data = "data/ptb.test"  # the output file for word to number

with codecs.open(vocab, "r", "utf-8") as f_vocab:
    words_id_dict = dict((w.strip(), idx) for idx, w in enumerate(f_vocab))


def get_id(word):
    """
    find the word id in words id dict

    Parameter
    ---------
    word: str, the word in corpus

    Return
    ------
    the id for input word
    """
    return words_id_dict[word] if word in words_id_dict else words_id_dict['<unk>']


f_in = codecs.open(raw_data, "r", encoding="utf-8")
f_out = codecs.open(output_data, "w", encoding="utf-8")

for line in f_in:
    sentence = line.strip().split() + ["eos"]
    out_line = " ".join([str(get_id(word)) for word in sentence]) + "\n"
    f_out.write(out_line)

f_in.close()
f_out.close()

