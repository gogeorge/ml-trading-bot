from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca_trade_api import REST
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from timedelta import Timedelta 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM, concatenate, Input, Flatten, Reshape, RepeatVector
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from alpaca.trading.client import TradingClient
from attention import Attention
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
from numpy import array_str
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
waitTime = 20
data = 0

current_position, current_price = 0, 0
predicted_price = 0

class stockPredALSTM:
    def __init__(self,
                 exchange: str = 'FTXU',
                 feature: str = 'close',

                 look_back: int = 50,
                 historical_period: int = 100,
                 neurons: int = 32,
                 activ_func: str = 'linear',
                 dropout: float = 0.15,
                 loss: str = 'mse',
                 optimizer: str = 'adam',
                 epochs: int = 20,
                 batch_size: int = 32,
                 output_size: int = 1
                 ):
        self.exchange = exchange
        self.feature = feature

        self.look_back = look_back
        self.historical_period = historical_period

        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size
        self.api = REST(base_url=config.BASE_URL, key_id=config.API_KEY, secret_key=config.API_SECRET)

    
    def log_output(self, pred_true, avg_mae, avg_mse, avg_rmse, avg_r2):
        with open('trials.txt', 'r') as file:
            contents = file.readlines()

        contents.append(
            '================= <Log - ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '> =================\n' +
            '\n' +
            '---------------- Hyperparameters ----------------\n' +
            # '(Attention layer enabled: 0.01, 0.01)' + '\n'
            'Epochs: ' + str(self.epochs) + '\n' +
            'Lookback: ' + str(self.look_back) + '\n' +
            'Historical Period: ' + str(self.historical_period) + '\n' +
            'Batch Size: ' + str(self.batch_size) + '\n' +
            'Neurons: ' + str(self.neurons) + '\n' +
            'Dropout rate: ' + str(self.dropout) + '\n' +
            '---------------- Trading Prices ----------------\n' +
            'Opening, Lowest, Highest Price (' + trading_pair + '): ' + array_str(pred_true) + '\n' +
            'Lowest-Highest Avg Price (' + trading_pair + '): ' + str((pred_true[0][1] + pred_true[0][2])/2) + '\n' + 
            'Current Price (' + trading_pair + '): ' + str(current_price) + '\n' +
            '---------------- Evaluation ----------------\n' +
            'Average MAE: ' + str(avg_mae) + '\n' +
            'Average MSE: ' + str(avg_mse) + '\n' +
            'Average RMSE: ' + str(avg_rmse) + '\n' +
            'Average R2 Score: ' + str(avg_r2) + '\n' +
            '---------------- Optional Evaluation ----------------\n' +
            'Opening, Lowest, Highest Price on ' + (date.today() + Timedelta(days=1)).strftime("%Y-%m-%d") + ' (next day): ' +
            '\n\n\n=== END LOG === \n\n\n\n'
        )

        with open('trials.txt', 'w') as file:
            file.writelines(contents)


    def getAllData(self):

        # Alpaca Market Data Client
        data_client = CryptoHistoricalDataClient()

        time_diff = datetime.now() - relativedelta(hours=self.historical_period)
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

    def getFeature(self, df, ft):
        data = df.filter([ft])
        data = data.values
        return data

    def scaleData(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def getTrainData(self, scaled_data, test_size=0.2):
        x, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            x.append(scaled_data[i:i+self.look_back])
            y.append(scaled_data[i + self.look_back])
        x, y = np.array(x), np.array(y)

        # Determine the size of the testing set
        split_index = int(len(x) * (1 - test_size))

        # Split the data into training and testing sets
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return x_train, y_train, x_test, y_test


    def evaluate_model_with_cv(self, model, x, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores, mse_scores, rmse_scores, r2_scores = [], [], [], []
        y_test = []

        for train_index, test_index in tscv.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred = np.concatenate(y_pred)
            y_pred = y_pred.flatten()
            y_test = np.concatenate(y_test)
            print(y_pred.shape, y_test.shape)
            # Reshape predictions and test data if necessary
            if len(y_pred.shape) > 1:
                y_pred = y_pred.squeeze()
            if len(y_test.shape) > 1:
                y_test = y_test.squeeze()

            # Calculate evaluation metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            mae_scores.append(mae)
            mse_scores.append(mse)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

        # Calculate average scores
        avg_mae = np.mean(mae_scores)
        avg_mse = np.mean(mse_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)

        return avg_mae, avg_mse, avg_rmse, avg_r2

    def visualize_predictions(self, y_true, y_pred):
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label='True Prices', color='blue')
        plt.plot(y_pred, label='Predicted Prices', color='red')
        plt.title('True vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    
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
        # Branch 1 - Predicting the opening price
        main_input = Input(shape=(input_data.shape[1], input_data.shape[2]), name='main_input')
        print('main_input: ', main_input.shape)

        branch_1_lstm_1 = LSTM(units=self.neurons, return_sequences=True)(main_input)
        branch_1_dropout_1 = Dropout(rate=self.dropout)(branch_1_lstm_1)
        branch_1_lstm_2 = LSTM(units=self.neurons)(branch_1_dropout_1)  # shape (None, 64)
        branch_1_dropout_2 = Dropout(rate=self.dropout)(branch_1_lstm_2) # shape (None, 64)
        # branch_1_attention = Attention(neurons=self.neurons, l1=0.01, l2=0.02)(branch_1_dropout_2)
        branch_1_output = Dense(units=1, name='branch_1_output')(branch_1_dropout_2)  # Opening price: shape (None, 1)

        branch_1_output_repeated = RepeatVector(self.look_back)(branch_1_output)
        # Branch 2 - Predicting the lowest price
        branch_2_lstm_start = LSTM(units=self.neurons, return_sequences=True)(main_input) # shape (None, 200, 64)
        branch_1_2_combined = concatenate([branch_1_output_repeated, branch_2_lstm_start]) # shape (None, 200, 65)
        print(branch_2_lstm_start.shape, branch_1_output_repeated.shape, branch_1_2_combined.shape)
        branch_2_lstm_1 = LSTM(units=self.neurons, return_sequences=True)(branch_1_2_combined)
        branch_2_dropout_1 = Dropout(rate=self.dropout)(branch_2_lstm_1)
        branch_2_lstm_2 = LSTM(units=self.neurons)(branch_2_dropout_1)
        branch_2_dropout_2 = Dropout(rate=self.dropout)(branch_2_lstm_2)
        # branch_2_attention = Attention(neurons=self.neurons, l1=0.01, l2=0.02)(branch_2_dropout_2)
        branch_2_output = Dense(units=1, name='branch_2_output')(branch_2_dropout_2)  # Lowest price

        # Branch 3 - Predicting the highest price
        branch_3_lstm_start = LSTM(units=self.neurons, return_sequences=True)(main_input)
        branch_final_combined = concatenate([branch_1_output_repeated, branch_3_lstm_start])
        branch_3_lstm_1 = LSTM(units=self.neurons, return_sequences=True)(branch_final_combined)
        branch_3_dropout_1 = Dropout(rate=self.dropout)(branch_3_lstm_1)
        branch_3_lstm_2 = LSTM(units=self.neurons)(branch_3_dropout_1)
        branch_3_dropout_2 = Dropout(rate=self.dropout)(branch_3_lstm_2)
        # branch_3_attention = Attention(neurons=self.neurons, l1=0.01, l2=0.02)(branch_3_dropout_2)
        branch_3_output = Dense(units=1, name='branch_3_output')(branch_3_dropout_2)  # Highest price

        # Creating the model
        model = Model(inputs=main_input, outputs=[branch_1_output, branch_2_output, branch_3_output])
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

        # can also do diff=open-close as a feature
        data_open = self.getFeature(df, self.feature)
        data_low = self.getFeature(df, 'low')
        data_high = self.getFeature(df, 'high')

        data = np.concatenate((data_open, data_low, data_high), axis=1)

        logger.info("Scaling Data")
        # scale data and get scaler
        scaled_data, scaler = self.scaleData(data)
        print('scaled shape: ', scaled_data.shape)
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
        x_pred = scaled_data[-self.look_back:].reshape(1, self.look_back, 3)

        # Predict the result
        logger.info("Predicting Price")
        pred = model.predict(x_pred)

        logger.info('------- Model Evaluation -------')
       
        avg_mae, avg_mse, avg_rmse, avg_r2 = self.evaluate_model_with_cv(model, x_train, y_train)

        bias = 1
        sentiment_threshold = 0.9
        if prob > sentiment_threshold:
            bias = 0.95
        else:
            bias = -0.5
        # fuse = (pred.item())*(prob**(1-bias))
        print('pred: ', pred)
        print('fuse')

        pred_list = []
        for p in pred:
            pred_list.append(p[0])

        pred_list = np.array(pred_list)
        pred_true = scaler.inverse_transform(pred_list.T)

        self.log_output(pred_true, avg_mae, avg_mse, avg_rmse, avg_r2)

        # Inverse the scaling to get the actual price
        return (pred_true[0][1] + pred_true[0][2]) / 2

