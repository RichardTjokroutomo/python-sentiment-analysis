import numpy as np
import random
import helper_func

def gradient_descent(rnn, data, word_idx, word_list_len, backprop=True, learning_rate=0.02):
    items = list(data.items())
    random.shuffle(items)

    correct_pred = 0

    for x, y in items:
        hotkey_x = helper_func.convert_to_hotkey(x, word_list_len, word_idx)
        y_parsed = int(y) # y is only T/F

        # feedforward
        rnn_y, dummy = rnn.feedforward(hotkey_x) # dummy is h, wont be used by us
        y_pred = helper_func.softmax(rnn_y)

        # check whether prediction is correct
        correct_pred += int(np.argmax(y_pred) == y_parsed)

        # get the nabla_loss function
        nabla_loss = y_pred
        nabla_loss[y_parsed] -= 1

        # backprop
        if backprop:
            rnn.backprop(nabla_loss, learning_rate)

    return correct_pred / len(data)

def gd(rnn, train_data, test_data, epoch, learning_rate, word_idx, num_of_words):
    for epoch in range(epoch):
        train_acc = gradient_descent(rnn, train_data, word_idx, num_of_words)

        if epoch % 100 == 99:
            print("epoch " + str(epoch+1) + " train accuracy: " + str(train_acc))

            test_acc = gradient_descent(rnn, test_data, word_idx, num_of_words, False)
            print("epoch " + str(epoch+1) + " test accuracy: " + str(test_acc))
            print("")