from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import logging
import asyncio
import config
from stock_prediction import stockPred
from stock_prediction_alstm import stockPredALSTM


# ENABLE LOGGING - options, DEBUG,INFO, WARNING?
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca Trading Client
trading_client = TradingClient(
    config.API_KEY, config.API_SECRET, paper=True)


# Trading variables
trading_pair = 'ETH/USD'
qty_to_trade = 1
# Wait time between each bar request and training model
waitTime = 100
data = 0

current_position, current_price = 0, 0
predicted_price = 0


async def main():
    '''
    Main function to get latest asset data and check possible trade conditions
    '''

    while True:
        logger.info('----------------------------------------------------')

        pred = stockPredALSTM()
        global predicted_price, current_price
        predicted_price = pred.predictModel()
        current_price = pred.getCurrentPrice()
        logger.info("Predicted Price is {0}".format(predicted_price))
        l1 = loop.create_task(check_condition())
        await asyncio.wait([l1])
        await asyncio.sleep(waitTime)



async def check_condition():
    global current_position, current_price, predicted_price
    current_position = get_positions()
    logger.info("Current Price is: {0}".format(current_price))
    logger.info("Current Position is: {0}".format(current_position))
    # If we do not have a position and current price is less than the predicted price place a market buy order
    if float(current_position) <= 0.01 and current_price < predicted_price:
        logger.info("Placing Buy Order")
        buy_order = await post_alpaca_order('buy', 1)
        if buy_order:
            logger.info("Buy Order Placed")
            print("Balance?: ", current_position)

    # If we do have a position and current price is greater than the predicted price place a market sell order
    if float(current_position) >= 0.01 and current_price > predicted_price:
        logger.info("Placing Sell Order")
        sell_order = await post_alpaca_order('sell', 1)
        if sell_order:
            logger.info("Sell Order Placed")
            print("Balance?: ", current_position)
    
    # if we have a position and the current price is really low buy again
    if float(current_position) >= 0.01 and (predicted_price - current_price)/predicted_price >= 0.1:
        logger.info("Placing Add. Buy Order")
        buy_order = await post_alpaca_order('buy', 0.5)
        if buy_order:
            logger.info("Buy Order Placed")
            print("Balance?: ", current_position)


async def post_alpaca_order(side, size):
    '''
    Post an order to Alpaca
    '''
    try:
        if side == 'buy':
            market_order_data = MarketOrderRequest(
                symbol="ETHUSD",
                qty=size - 0.2,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            buy_order = trading_client.submit_order(
                order_data=market_order_data
            )
            return buy_order
        else:
            market_order_data = MarketOrderRequest(
                symbol="ETHUSD",
                qty=size - 0.2,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            sell_order = trading_client.submit_order(
                order_data=market_order_data
            )
            return sell_order
    except Exception as e:
        logger.exception(
            "There was an issue posting order to Alpaca: {0}".format(e))
        return False


def get_positions():
    positions = trading_client.get_all_positions()
    # print(positions[0])
    global current_position
    for p in positions:
        if p.symbol == 'ETHUSD':
            current_position = p.qty
            return current_position
    return current_position


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()