# ML Trading Bot


## Description

This is a trading bot that uses an Attention-LSTM model, that is a Long Term Short Term Memory model with a custom Attention layer attached to it in order to learn how a currency is behaving and predict the future price of that currency. This model is then fused with a sentiment model that rates the positivity or negativity of the crypto-related news. The sentiment model affects the final prediction way less than the LSTM model but that can be adjusted through a bias in the fuse equation.

## Installation

The bot works via the Alpaca API and thus the API-Key and API-Secret are needed. 

To run the bot use:

``` python main.py ```


## Read more

Current inspiration for the architecture of the bot came from the following articles:

https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf \\
https://machinelearningmastery.com/the-attention-mechanism-from-scratch/