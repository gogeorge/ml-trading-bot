# ML Trading Bot


## Description

This is a trading bot that uses two types of LSTM models (Long Term Short Term Memory): 
  1. LSTM model with a custom Attention layer attached to it in order to predict the closing price of a crypto-currency.
  2. A multi-value assosicate LSTM model for closing, lowest and highest price.

Furthermore, these models can be fused with a sentiment model that rates the positivity or negativity of the crypto-related news. The sentiment model affects the final prediction way less than the LSTM model but that can be adjusted through a bias in the fuse equation. The use of training and using these models on a crypto currency was to be able to study the patterns quicker than with normal stocks. 

To changed the type of model being used, you can switch between ```stockPred``` (LSTM with Attention) and ```stockPredALSTM``` (Assosiciate LSTM).


## Installation

The bot works via the Alpaca API and thus the API-Key and API-Secret are needed. 

To run the bot use:

``` python main.py ```


## Read more

Current inspiration for the architecture of the bot came from the following articles:

https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf \\
https://machinelearningmastery.com/the-attention-mechanism-from-scratch/
