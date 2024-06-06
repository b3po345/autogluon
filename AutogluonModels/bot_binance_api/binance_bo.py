import logging
import sqlite3
import pandas as pd
from binance.client import Client
import requests
import time
from autogluon.tabular import TabularPredictor
from autogluon.core.searcher import EvolutionaryAlgorithmSearchCV
import sys

class FitnessFunction:
    @staticmethod
    def calculate_fitness(params):
        buy_threshold = params[0]
        sell_threshold = params[1]
        stop_loss_threshold = params[2]
        commission_rate = params[3]
        minimum_profit_rate = params[4]
        
        current_balance = 200
        amount = 10
        total_trades = 0
        total_profit = 0
        
        while current_balance > 10 and total_trades < 20:
            simulated_data = generate_simulated_data()
            
            if should_buy(simulated_data, buy_threshold):
                buy_amount = calculate_buy_amount(simulated_data, current_balance, amount)
                current_balance -= buy_amount
                total_trades += 1
            
            elif should_sell(simulated_data, sell_threshold):
                sell_amount = calculate_sell_amount(simulated_data, amount)
                current_balance += sell_amount
                total_trades += 1
                
            elif should_stop_loss(simulated_data, stop_loss_threshold):
                pass
            
            total_profit = calculate_total_profit(current_balance)
        
        return total_profit

def initialize_binance_api(api_key, api_secret):
    """
    Инициализация клиента Binance API
    :param api_key: Ключ API
    :param api_secret: Секрет API
    :return: Объект клиента Binance API
    """
    client = Client(api_key, api_secret)
    return client

def get_account_info(client):
    """
    Получение информации о счете пользователя
    :param client: Объект клиента Binance API
    :return: Информация о счете пользователя
    """
    account_info = client.get_account()
    return account_info

def get_exchange_info(client):
    """
    Получение информации о бирже
    :param client: Объект клиента Binance API
    :return: Информация о бирже
    """
    exchange_info = client.get_exchange_info()
    return exchange_info

def extract_data_from_database(db_path, query):
    """
    Извлечение данных из базы данных
    :param db_path: Путь к базе данных
    :param query: SQL-запрос для извлечения данных
    :return: Данные о котировках
    """
    conn = sqlite3.connect(db_path)
    klines_new = pd.read_sql_query(query, conn)
    conn.close()
    return klines_new

def train_model(data):
    """
    Обучение модели на основе данных
    :param data: Данные для обучения
    :return: Обученная модель
    """
    predictor = TabularPredictor(label='close', path='models').fit(data)
    return predictor

def make_predictions(model, data):
    """
    Создание предсказаний на основе модели
    :param model: Обученная модель
    :param data: Данные для предсказания
    :return: Предсказания
    """
    predictions = model.predict(data)
    predictions_df = pd.DataFrame({'prediction': predictions}, index=data.index)
    return predictions_df

def get_symbols_with_USDT():
    try:
        response = requests.get('https://api.binance.com/api/v3/exchangeInfo')
        response.raise_for_status()
        exchange_info = response.json()

        symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['quoteAsset'] == 'USDT']
        return symbols
    except requests.exceptions.RequestException as e:
        logging.error("Error fetching symbols: %s", e)
        return []

def update_balance_after_trade(trade_type, quantity, price, commission_rate, current_balance, symbol=None):
    if trade_type == 'buy':
        current_balance -= quantity * price * (1 + commission_rate)
    elif trade_type == 'sell':
        current_balance += quantity * price * (1 - commission_rate)
    return current_balance

def get_new_data(api_key, api_secret):
    try:
        client = Client(api_key, api_secret)
        symbols = get_symbols_with_USDT()
        all_data = []

        for symbol in symbols:
            new_data = client.get_historical_klines(symbol=symbol, interval='1h', limit=100)
            new_data_df = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            all_data.append(new_data_df)

        return pd.concat(all_data, ignore_index=True)

    except Exception as e:
        logging.error("Error fetching new data: %s", e)
        return None

def execute_trades(predictions, number_of_trades, buy_threshold, sell_threshold, stop_loss_threshold, commission_rate, current_balance, investment_amount, client, model, minimum_profit_rate, api_key, api_secret):
    trades_executed = 0
    peak_price = 0
    symbols_checked = False
    
    for index, row in predictions.iterrows():
        current_price = row['prediction']
        
        if trades_executed < number_of_trades:
            if current_price * (1 + buy_threshold) <= current_balance:
                try:
                    if not symbols_checked:
                        symbols = get_symbols_with_USDT()
                        symbols_checked = True
                    
                    if symbols:
                        for symbol in symbols:
                            logging.info(f"Checking before buy {symbol}: Sufficient funds available.")
                            order = client.create_order(
                                symbol=symbol,
                                side='BUY',
                                type='LIMIT',
                                quantity=investment_amount / current_price,
                                price=current_price
                            )
                            logging.info("Buy order executed: %s", order)
                            
                            trades_executed += 1
                            peak_price = current_price
                            
                            current_balance = update_balance_after_trade('buy', investment_amount / current_price, current_price, commission_rate, current_balance)
                            
                            new_data = get_new_data(api_key, api_secret)
                            model = train_model(new_data)
                
                except Exception as e:
                    logging.error("Error executing buy: %s", e)
        
        if current_price > peak_price * (1 + sell_threshold) or current_price < peak_price * (1 - stop_loss_threshold):
            try:
                if quantity > 0:
                    logging.info("Checking before sell: Sufficient assets available.")
                    
                    if symbols:
                        for symbol in symbols:
                            logging.info(f"Checking before sell {symbol}: Sufficient assets available.")
                            order = client.create_order(
                                symbol=symbol,
                                side='SELL',
                                type='LIMIT',
                                quantity=investment_amount / current_price,
                                price=current_price
                            )
                            logging.info("Sell order executed: %s", order)
                            
                            if ((current_price - peak_price) / peak_price) * 100 >= minimum_profit_rate:
                                peak_price = current_price
                                
                                current_balance = update_balance_after_trade('sell', investment_amount / current_price, current_price, commission_rate, current_balance)
                                
                                new_data = get_new_data(api_key, api_secret)
                                model = train_model(new_data)
                            else:
                                logging.info("Not selling: Profit is less than minimum desired profit.")
                
            except Exception as e:
                logging.error("Error executing sell: %s", e)
    
    return trades_executed, current_balance


# Определение вашей функции fitness_function без импорта GeneticAlgorithm
def fitness_function(params):
    """
    Функция оценки пригодности для генетического алгоритма
    :param params: Параметры для оценки
    :return: Значение функции оценки
    """
    buy_threshold = params[0]
    sell_threshold = params[1]
    stop_loss_threshold = params[2]
    commission_rate = params[3]
    minimum_profit_rate = params[4]
    
    # Симуляция торговой стратегии с заданными параметрами
    current_balance = 200
    amount = 10
    total_trades = 0
    total_profit = 0
    
    while current_balance > 10 and total_trades < 20:
        # Генерация случайных данных для симуляции торгов
        simulated_data = generate_simulated_data()
        
        # Симулируем стратегию покупки и продажи активов
        if should_buy(simulated_data, buy_threshold):
            buy_amount = calculate_buy_amount(simulated_data, current_balance, amount)
            current_balance -= buy_amount
            total_trades += 1
            # Здесь могут быть другие операции, связанные с покупкой
            
        elif should_sell(simulated_data, sell_threshold):
            sell_amount = calculate_sell_amount(simulated_data, amount)
            current_balance += sell_amount
            total_trades += 1
            # Здесь могут быть другие операции, связанные с продажей
            
        elif should_stop_loss(simulated_data, stop_loss_threshold):
            # Здесь могут быть операции, связанные со стоп-лоссом
        
           total_profit = calculate_total_profit(current_balance)
    
    # Возвращаем значение пригодности (например, общую прибыль)
    return total_profit


def main():
    api_key = 'eIzZMMK4v1SUkL4MiiRAWxoVb5OdbbpFJhENhNQIClXGSNhzJN4qFcRm1Hr3yRr3'
    api_secret = 'PQLAIvj87NOt8OMQLxZVUX3goV75gACU0OBDj7QkQ6IvfJjuxPKCKXUWDg4M0K3o'
    client = initialize_binance_api(api_key, api_secret)

    account_info = get_account_info(client)
    logging.info("Account information: %s", account_info)

    exchange_info = get_exchange_info(client)
    logging.info("Exchange information: %s", exchange_info)

    db_path = 'C:\\Program Files\\DB Browser for SQLite\\binance_data.db'

    query = "SELECT close_time, open, high, low, close, volume, symbol FROM binance_prices ORDER BY close_time ASC"

    while True:
        klines_new = extract_data_from_database(db_path, query)

        logging.info("Quote data: %s", klines_new)

        model = train_model(klines_new)

        predictions = make_predictions(model, klines_new)

        global number_of_trades, buy_threshold, sell_threshold, stop_loss_threshold, commission_rate, current_balance, investment_amount, minimum_profit_rate

        callable_fitness_function = FitnessFunction().calculate_fitness
        genetic_algorithm = EvolutionaryAlgorithmSearchCV(100, 0.1, callable_fitness_function, 6)  
        best_params = genetic_algorithm.fit(params, 'real')
        buy_threshold, sell_threshold, stop_loss_threshold, commission_rate, investment_amount, minimum_profit_rate = best_params

        trades_executed, current_balance = execute_trades(predictions, number_of_trades, buy_threshold, sell_threshold,
                                                          stop_loss_threshold, commission_rate, current_balance,
                                                          investment_amount, client, model, minimum_profit_rate, api_key, api_secret)

        logging.info("Trading completed. Current balance: %s, Trades executed: %s", current_balance, trades_executed)

        time.sleep(3600)

if __name__ == "__main__":
    logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
