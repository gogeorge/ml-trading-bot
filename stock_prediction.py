from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca_trade_api import REST
from datetime import datetime
from dateutil.relativedelta import relativedelta
from timedelta import Timedelta 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from alpaca.trading.client import TradingClient
from attention import Attention
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from finbert import estimate_sentiment
import config
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

trading_client = TradingClient(
    config.API_KEY, config.API_SECRET, paper=True)

trading_pair = 'ETH/USD'
qty_to_trade = 1
# Wait time between each bar request and training model
waitTime = 100
data = 0

current_position, current_price = 0, 0
predicted_price = 0

class stockPred:
    def __init__(self,
                 exchange: str = 'FTXU',
                 feature: str = 'close',

                 look_back: int = 100,

                 neurons: int = 50,
                 activ_func: str = 'linear',
                 dropout: float = 0.2,
                 loss: str = 'mse',
                 optimizer: str = 'adam',
                 epochs: int = 25,
                 batch_size: int = 32,
                 output_size: int = 1
                 ):
        self.exchange = exchange
        self.feature = feature

        self.look_back = look_back

        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size
        self.api = REST(base_url=config.BASE_URL, key_id=config.API_KEY, secret_key=config.API_SECRET)


    def getAllData(self):

        # Alpaca Market Data Client
        data_client = CryptoHistoricalDataClient()

        time_diff = datetime.now() - relativedelta(hours=3000)
        logger.info("Getting bar data for {0} starting from {1}".format(
            trading_pair, time_diff))
        # Defining Bar data request parameters
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[trading_pair],
            timeframe=TimeFrame.Hour,
            start=time_diff
        )
        # Get the bar data from Alpaca
        df = data_client.get_crypto_bars(request_params).df
        global current_price
        current_price = df.iloc[-1][self.feature]
        return df

    @staticmethod
    def getCurrentPrice():
        return current_price

    def getFeature(self, df):
        data = df.filter([self.feature])
        data = data.values
        return data

    def scaleData(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def getTrainData(self, scaled_data, test_size=0.2):
        x, y = [], []
        for price in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[price - self.look_back:price, :])
            y.append(scaled_data[price, :])
        x, y = np.array(x), np.array(y)

        # Determine the size of the testing set
        split_index = int(len(x) * (1 - test_size))

        # Split the data into training and testing sets
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return x_train, y_train, x_test, y_test
    
    def get_evaluation(self, x_test, y_test, model, scaler):
        print("Shapes of x_test and y_test:", x_test.shape, y_test.shape)
        # Predict on the testing set
        y_pred = model.predict(x_test)

        # Inverse transform the predicted and true values if you scaled them
        y_pred_inverse = scaler.inverse_transform(y_pred)
        y_test_inverse = scaler.inverse_transform(y_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
        mse = mean_squared_error(y_test_inverse, y_pred_inverse)
        rmse = mean_squared_error(y_test_inverse, y_pred_inverse, squared=False)
        r2 = r2_score(y_test_inverse, y_pred_inverse)
        logger.info(f"Mean Absolute Error (MAE): {mae}")
        logger.info(f"Mean Squared Error (MSE): {mse}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse}")
        logger.info(f"R-squared (R2) Score: {r2}")

    
    def get_dates(self): 
        today = datetime.now()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')
    
    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=trading_pair, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        print(news)
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 

    def LSTM_model(self, input_data):
        model = Sequential()
        model.add(LSTM(self.neurons, input_shape=(
            input_data.shape[1], input_data.shape[2]), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons))
        model.add(Dropout(self.dropout))
        model.add(Attention(self.neurons, regularization=regularizers.l2(0.01)))
        model.add(Dense(units=self.output_size))
        model.add(Activation(self.activ_func))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def trainModel(self, x, y):
        x_train = x[: len(x) - 1]
        y_train = y[: len(x) - 1]
        print('shape, ', x_train.shape[1], x_train.shape[2])
        model = self.LSTM_model(x_train)
        modelfit = model.fit(x_train, y_train, epochs=self.epochs,
                             batch_size=self.batch_size, verbose=1, shuffle=True)
        return model, modelfit

    def predictModel(self):
        logger.info("Getting Ethereum Bar Data")
        # get all data
        df = self.getAllData()

        logger.info("Getting Feature: {}".format(self.feature))
        # get feature (closing price)
        data = self.getFeature(df)

        logger.info("Scaling Data")
        # scale data and get scaler
        scaled_data, scaler = self.scaleData(data)

        logger.info("Getting Train Data")
        # get train data
        x_train, y_train, x_test, y_test = self.getTrainData(scaled_data)

        logger.info("Training Model")
        # Creates and returns a trained model
        model = self.trainModel(x_train, y_train)[0]

        logger.info("Getting News Data")

        prob, sentiment = self.get_sentiment()
        logger.info(prob)
        logger.info(sentiment)

        logger.info("Extracting data to predict on")
        x_pred = scaled_data[-self.look_back:].reshape(1, self.look_back, 1)

        # Predict the result
        logger.info("Predicting Price")
        pred = model.predict(x_pred)

        logger.info('------- Model Evaluation -------')
        self.get_evaluation(x_test, y_test, model, scaler)
        # fuse
        bias = 1
        sentiment_threshold = 0.9
        if prob > sentiment_threshold:
            bias = 0.95
        else:
            bias = -0.5
        fuse = (pred.item())*(prob**(1-bias))
        print('fuse')
        print(fuse)
        pred = fuse.squeeze()
        pred = np.array([float(pred)])
        pred = np.reshape(pred, (pred.shape[0], 1))
        print('pred rescale')
        print(pred)

        # Inverse the scaling to get the actual price
        pred_true = scaler.inverse_transform(pred)
        return pred_true[0][0]
