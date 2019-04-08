#! /usr/bin/python
# -*- coding: utf-8 -*-
# @ FileName    : vocab.py
# @ CreatedDate : 2019-04-07 18:18
# @ Author      : Joshua Chan


import codecs
import collections
from operator import itemgetter

raw_data = "data/ptb.train.txt"     # train data
vocab_output = "data/ptb.vocab"

counter = collections.Counter()

with codecs.open(raw_data, "r", encoding="utf-8") as f_in:
    for line in f_in:
        for word in line.strip().split():
            counter[word] += 1

# according the frequency of words to sort words, by descending
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)

sorted_words = [x[0] for x in sorted_word_to_cnt]

# we need to add end flag for each sentence
sorted_words = ["<eos>"] + sorted_words

# for other corpus you need to translate some low frequency words into <uniq>
# sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
# if len(sorted_words) > 10000:
#     sorted_word = sorted_words[:10000]


with codecs.open(vocab_output, "w", "utf-8") as f_out:
    for word in sorted_words:
        f_out.write(word + "\n")


