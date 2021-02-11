# LSTM model for time series prediction

A very simple and lightweight long short-term memory neural network model using PyTorch wrapped to a function to get predictions for time series data and some testing with randomised data. It predicts surprisingly well especially randomised sine waves considering how simple the model is.

Considering how simple the network is, it predicts surprisingly well, with few caveats though. 
Firstly, the example time series are not random walk, instead some noisified continuous functions, mostly sinewaves that should be pretty easy to predict, anyways.
For non-stationary random time series the model obviously would not work any better than random guesses.

Secondly, the LSTM network does not catch linear trend very well. This is because the LSTM tends to give predictions that fall pretty close the values of the training data. Thus the linear trends would need to be removed before feeding to the network.

## How and why it works?

The long short-term memory network first introduced in 1997 [1] is an extension of recurrent neural networks that have a feedback loop that lets them get a memory of history. The main limitation with RNNs is that they only catch short dependencies as the older inputs are gradually given forgotten and the network fails to catch long-term dependencies [2]. That is the problem what LSTM solved.

In essence, the vanilla LSTM network has a cell state that contains all relevant information. The relevance of old information is determined in forget gates and multiplied to the old state of the LSTM. Then, the relevance of new information is determined input gate which is multiplied with tanh layer that gives candidate values to be added to the old state. The result is then added to the old state which has irrelevant old information removed by forget gate values. The result is the next cell state of the LSTM.

The output is determined by the new cell state which is normalised between [-1, 1] by tanh function and multiplied by sigmoid gate which decides which parts of the state is outputted. The vanilla LSTM network has been modified in a number of interesting ways, see [3] for an interesting comparison of the variants.

For time series predictions, the LSTM learns from past datapoints the information it determines relevant. Thus it is able to capture patterns from past data, and forget random noise, assuming the historical data can be used to predict future values.



[1]: Hochreiter, Schmidhuber, Long short-term memory. Neural computation, 1997

[2]: Bengio et al., Learning long-term dependencies with gradient descent is difficult. IEEE TNN, 1994

[3]: Greff et al., LSTM: A search space odyssey. IEEE TNNLS, 2016
