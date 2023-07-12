# Python Sentiment Analysis

This project uses RNN for sentiment analysis using [Victor Zhou's](https://github.com/vzhou842/rnn-from-scratch/blob/master/data.py) custom dataset.


 This project is a code-along from Victor Zhou's [blog](https://victorzhou.com/blog/intro-to-rnns/), although the final result is not the same as I modified the code.


Details:
- number of layers: 1
- activation function: tanh
- softmax is applied to the output of the RNN
- loss function: cross entropy loss
- epoch: 1000 (configurable)
- learning rate: 0.02 (configurable)

### Getting started
first, install all the modules from requirements:
```
pip install -r requirements.txt
```
### Running the code
```
Python3 master.py
```