import os
import pandas as pd
import logging
import time
import ta
from sklearn.preprocessing import StandardScaler
from binance.client import Client
from autogluon.tabular import TabularPredictor
from autogluon.core.scheduler import FIFOScheduler
from autogluon.core import Real, Categorical, Integer
import sqlite3
import logging.config

logging.config.fileConfig('C:\\Program Files\\Python39\\AB project\\logging.ini')

# Инициализация клиента Binance
api_key = 'eIzZMMK4v1SUkL4MiiRAWxoVb5OdbbpFJhENhNQIClXGSNhzJN4qFcRm1Hr3yRr3'
api_secret = 'PQLAIvj87NOt8OMQLxZVUX3goV75gACU0OBDj7QkQ6IvfJjuxPKCKXUWDg4M0K3o'
client = Client(api_key, api_secret)

# Функция для получения всех торговых пар
def get_symbols_for_quote_assets(allowed_quote_assets):
    all_symbols_data = client.get_exchange_info()
    trading_pairs = []
    for symbol in all_symbols_data['symbols']:
        if symbol['status'] == 'TRADING' and symbol['quoteAsset'] in allowed_quote_assets:
            trading_pairs.append(symbol['symbol'])
    return trading_pairs

# Функция для создания подключения к базе данных
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('C:/Program Files/DB Browser for SQLite/binance_data.db')
        print("Подключение к базе данных успешно.")
        return conn
    except sqlite3.Error as e:
        print(f"Ошибка при подключении к базе данных: {e}")
    return conn

# Функция для создания таблицы
def create_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS binance_prices (
                          timestamp DATETIME,
                          symbol TEXT,
                          open REAL,
                          high REAL,
                          low REAL,
                          close REAL,
                          volume REAL,
                          PRIMARY KEY (timestamp, symbol)
                        )''')
        conn.commit()
    except Exception as e:
        logging.error(f"Произошла ошибка при создании таблицы: {e}")

def preprocess_data(df):
    try:
        existing_columns = set(df.columns)
        columns_to_drop = ['close_time', 'quote_asset_volume', 'number_of_trades',
                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

        df.drop(columns=columns_to_drop, inplace=True)

        if df.empty:
            logging.warning("DataFrame пуст после удаления столбцов.")
            return df

        logging.info(f"Типы данных в DataFrame:\n{df.dtypes}")
        logging.info(f"Уникальные значения в столбце 'symbol': {df['symbol'].unique()}")

        scaler = StandardScaler()
        features_to_scale = ['open', 'high', 'low', 'close', 'volume']
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

        return df
    except Exception as e:
        logging.error(f"Произошла ошибка во время предварительной обработки данных: {e}")
        return pd.DataFrame()
    

def load_data_from_binance(symbols, interval, limit, conn):
    try:
        data_frames = []
        for symbol in symbols:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                  'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                  'taker_buy_quote_asset_volume', 'ignore'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)

            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

            data['symbol'] = symbol
            logging.info(f"Data types in DataFrame for symbol '{symbol}':\n{data.dtypes}")

            data_frames.append(data)

        df_concatenated = pd.concat(data_frames)
        df_concatenated.to_sql('binance_prices', conn, if_exists='replace', index=False)

        return df_concatenated

    except Exception as e:
        logging.error(f"Error in loading data from Binance: {str(e)}")
        return None

def train_autogluon_model(symbols, interval, limit, num_bagging_folds=16, auto_stack="gpu", mem_limit=10):
    try:
        start_total_time = time.time()
        logging.info("Training started...")

        conn = create_connection()
        if conn is None:
            return None, None
        create_table(conn)

        data = load_data_from_binance(symbols, interval, limit, conn)
        if data is None:
            return None, None
        logging.info("Data successfully loaded.")

        start_preprocess_time = time.time()
        preprocessed_data = preprocess_data(data)
        if preprocessed_data is None or preprocessed_data.empty:
            return None, None
        logging.info("Data preprocessed successfully.")
        preprocess_duration = time.time() - start_preprocess_time
        logging.info(f"Data preprocessing time: {preprocess_duration} seconds.")

        label = preprocessed_data.iloc[:, -1]

        start_model_time = time.time()

        # Model training
        search_space = {'NN': {'num_epochs': Integer(5, 100)},
                        'GBM': {'num_boost_round': Integer(5, 100)}}

        predictor = TabularPredictor(label='symbol', problem_type='multiclass', eval_metric='accuracy')

        predictor.fit(train_data=preprocessed_data,
                       hyperparameters={'search_space': search_space},
                       scheduler_options={'resource': {'num_cpus': 4, 'num_gpus': 1},
                                          'searcher': 'random',
                                          'max_t': 100,
                                          'grace_period': 20})
        logging.info("Model trained successfully.")

        model_duration = time.time() - start_model_time
        logging.info(f"Model training time: {model_duration} seconds.")

        total_duration = time.time() - start_total_time
        logging.info(f"Total execution time: {total_duration} seconds.")

        # Saving the model
        predictor.save("autogluon_model")

        return preprocessed_data, predictor

    except Exception as e:
        logging.error(f"Error in model training process: {str(e)}")
        return None, None

def load_models(models_dir):
    """
    Загрузка моделей из указанного каталога.

    Parameters:
    models_dir (str): Путь к каталогу, содержащему модели.

    Returns:
    dict: Словарь, где ключами являются имена моделей, а значениями - сами модели.
    """
    models = {}  
    try:
        if os.path.exists(models_dir):
            model_files = [filename for filename in os.listdir(models_dir) if filename.endswith('.model')]
            for filename in tqdm(model_files, desc="Loading models"):
                model_path = os.path.join(models_dir, filename)
                model = TabularPredictor.load(model_path)
                model_name = filename.split('.')[0]
                models[model_name] = model
        return models
    except Exception as e:
        logging.error(f"Ошибка при загрузке моделей: {e}")
        return None
    
def main():
    allowed_quote_assets = ['USDT']
    symbols = get_symbols_for_quote_assets(allowed_quote_assets)  # Получить все доступные символы для указанных котируемых валют

    interval = "1m"  # Подставьте соответствующий интервал
    limit = 2000  # Установите нужное значение лимита

    preprocessed_df, predictor = train_autogluon_model(symbols, interval=interval, limit=limit, num_bagging_folds=16, auto_stack="gpu", mem_limit=10)

    if preprocessed_df is not None and predictor is not None:
        # Загрузка модели
        models_dir = "autogluon_model"
        predictor = load_models(models_dir)
        if predictor is None:
            print("Ошибка при загрузке моделей.")
            return

        try:
            # Получение новых данных для прогнозирования
            file_paths = ['D:/Admin/Desktop/бот2/Logs']  # Список путей к вашим файлам txt
            new_data = pd.DataFrame()  # Создание пустого DataFrame для объединения данных

            for file_path in file_paths:
                try:
                    # Чтение данных из файла с учётом ошибочных строк
                    temp_data = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)
                    # Объединение данных
                    new_data = pd.concat([new_data, temp_data], ignore_index=True)
                except pd.errors.ParserError as e:
                    print(f"Ошибка при чтении файла '{file_path}': {e}")

            if not new_data.empty:
                # Сохранение объединенных данных в новый файл в формате CSV
                new_data.to_csv('data_for_autogluon.csv', index=False)

                # Предварительная обработка данных
                new_data = preprocess_data_for_prediction(new_data)
                if new_data is None:
                    print("Ошибка при предварительной обработке новых данных для прогнозирования.")
                    return

                # Прогнозирование
                predictions = make_predictions(new_data, predictor)
                if predictions is None:
                    print("Ошибка при прогнозировании на основе новых данных.")
                    return

                # Визуализация результатов прогнозирования
                visualize_data_and_predictions(new_data, predictions)

            else:
                print("Нет данных для обработки.")
                # Дальнейшие действия в случае отсутствия данных

        except pd.errors.ParserError as e:
            print("Ошибка при чтении CSV-файла:", e)
            # Дальнейшие действия в случае ошибки чтения CSV-файла

    else:
        print("Во время обучения модели произошла ошибка. Проверьте журналы для получения дополнительной информации.")

if __name__ == '__main__':
    main()
