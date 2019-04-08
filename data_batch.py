#! /usr/bin/python
# -*- coding: utf-8 -*-
# @ FileName    : data_batch.py
# @ CreatedDate : 2019-04-07 19:18
# @ Author      : Joshua Chan
# @ Company     : PingAn OneConnect

import os
import numpy as np
import tensorflow as tf

train_data = "data/ptb.train"
train_batch_size = 20
train_num_step = 35


def read_data(file_path):
    """
    load corpus, transfer word idx into a list

    Parameter
    ---------
    file_path: str, the word id corpus path

    Return
    ------
    List[int]

    Raise
    -----
    IOError, depend on whether the file path exists
    if exists, then pass, otherwise, raise IOError

    """

    if not os.path.exists(file_path):
        raise IOError(f"{file_path} does not exist, please check out")

    with open(file_path, "r") as f_in:
        id_string = " ".join([line.strip() for line in f_in.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batch(id_list, batch_size, num_step):
    """
    According to the train data batch_size and num_step to make input data
    as train batch

    Parameters
    ----------
    id_list: int of list, a list contains word index
    batch_size: int, the train batch data size
    num_step: int, the rnn time step

    Return
    ------
    Batch data list
    """
    num_batchs = (len(id_list) - 1) // (batch_size * num_step)      # compute the number of batchs
    data = np.array(id_list[: num_batchs * batch_size * num_step])       # generate train data list

    # transform train data list to a matrix, batch_size x num_batchs*num_step
    data = np.reshape(data, [batch_size, num_batchs * num_step])

    # split data as a list contains a bunch of arrays which shape is batch_size x num_step
    data_batches = np.split(data, num_batchs, axis=1)

    # for label data, processing same as data
    label = np.array(id_list[1: num_batchs * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batchs * num_step])
    label_batches = np.split(label, num_batchs, axis=1)

    return list(zip(data_batches, label_batches))


if __name__ == "__main__":
    pass