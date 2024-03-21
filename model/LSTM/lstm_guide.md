# LSTM
长短时记忆

Before we start learning lstm, we first need to learn about RNN (循环神经网络). RNN is a type of 
neural network that is used to process sequential data. It is based on the idea that the output 
of the previous step is used as input for the current step.

In RNN, we will lose information while training the model. This is because of the vanishing gradient 
problem. To solve this problem, we use LSTM since it can keep the important information for a long time. 
Through the gates, LSTM can decide which information to keep and which to discard.

BRNN 双向循环神经网络，同时考虑过去和未来的信息
DRNN 深层循环神经网络，叠加多层RNN