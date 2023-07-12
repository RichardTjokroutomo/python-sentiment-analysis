import numpy as np

class RNN:

    def __init__(self, input_len, output_len, h_len):
        self.weights_yh = np.random.randn(output_len, h_len) / 1000 
        self.weights_hh = np.random.randn(h_len, h_len) / 1000
        self.weights_hx = np.random.randn(h_len, input_len) / 1000

        self.biases_y = np.random.randn(output_len, 1)
        self.biases_h = np.random.randn(h_len, 1)
    

    def feedforward(self, inputs):
        h = np.zeros((self.weights_hh.shape[0], 1)) # .shape returns (x, y). We only want x

        self.inputs = inputs
        self.hs = {0: h} 

        for i, x in enumerate(inputs):
            first_term = np.matmul(self.weights_hx, x)
            second_term = np.matmul(self.weights_hh, h)
            h = np.tanh(first_term + second_term + self.biases_h)
            self.hs[i+1] = h

        y = np.matmul(self.weights_yh, h) + self.biases_y

        return (y, h)

    def backprop(self, nabla_loss, learning_rate=0.02):

        rnn_len = len(self.inputs)

        # find the nablas for the arrow leading to the output
        nabla_weights_yh = nabla_loss @self.hs[rnn_len].T
        nabla_biases_y = nabla_loss

        # find the nablas for the arrows leading to h
        nabla_weights_hh = np.zeros_like(self.weights_hh)
        nabla_weights_hx = np.zeros_like(self.weights_hx)
        nabla_biases_h = np.zeros_like(self.biases_h)

        # dL/dh = dL/dy * dy/dh
        nabla_l_h = np.matmul(self.weights_yh.T, nabla_loss)

        for i in reversed(range(rnn_len)):

            # temp value dL/dh * (1 - h^2)
            temp = ((1-self.hs[i+1]**2) * nabla_l_h)

            nabla_biases_h += temp

            nabla_weights_hh += np.matmul(temp, self.hs[i].T)

            nabla_weights_hx += np.matmul(temp, self.inputs[i].T)

            # next dL/dh
            nabla_l_h = np.matmul(self.weights_hh, temp)

        # limits the nablas in [-1, 1]
        for i in [nabla_biases_h, nabla_biases_y, nabla_weights_hh, nabla_weights_hx, nabla_weights_yh]:
            np.clip(i, -1, 1, out=i)

        # update
        self.biases_h -= learning_rate*nabla_biases_h
        self.biases_y -= learning_rate*nabla_biases_y
        self.weights_hh -= learning_rate*nabla_weights_hh
        self.weights_hx -= learning_rate*nabla_weights_hx
        self.weights_yh -= learning_rate*nabla_weights_yh

