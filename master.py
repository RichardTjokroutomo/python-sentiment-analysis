
import helper_func
from rnn import RNN
import optimizers


# load the data
(train_data, test_data) = helper_func.get_data()
(words_list, num_of_words) = helper_func.get_vocab_list()

# assign indices
word_idx = helper_func.word_to_idx(words_list)
idx_word = helper_func.idx_to_word(words_list)


# setting up RNN
rnn = RNN(num_of_words, 2, 64)

# training RNN
optimizers.gd(rnn, train_data, test_data, 1000, 0.02, word_idx, num_of_words) # model, train, test, epoch, lr, word->idx dict, num of unique words


