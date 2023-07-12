import numpy as np
import data

def get_data():
    return (data.train_data, data.test_data)

def get_vocab_list():
    word_list = list(set([w for text in data.train_data.keys() for w in text.split(' ')]))
    word_list_len = len(word_list)

    return (word_list, word_list_len)

def word_to_idx(word_list):
    idx_list = {w: i for i, w in enumerate(word_list)}

    return idx_list

def idx_to_word(idx_list):
    word_list = {i: w for i, w in enumerate(idx_list)}

    return word_list

def convert_to_hotkey(text, word_list_len, word_idx_dict):
    hotkey = []
    for w in text.split(' '):
        v = np.zeros((word_list_len, 1))
        v[word_idx_dict[w]] = 1
        hotkey.append(v)

    return hotkey

def softmax(z):
  return np.exp(z) / sum(np.exp(z))
