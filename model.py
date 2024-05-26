import os
import pandas as pd
import logging
import time
import ta
from sklearn.preprocessing import StandardScaler
from binance.client import Client
from autogluon.tabular import TabularPredictor
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

        return df_concaten

    
def train_autogluon_model(symbols, interval, limit, num_bagging_folds=16, auto_stack="gpu", mem_limit=10)
    try:
        start_total_time = time.time()
        logging.info("Начало обучения...")

        conn = create_connection()
        if conn is None:
            return None, None
        create_table(conn)

        data = load_data_from_binance(symbols, interval, limit, conn)
        if data is None:
            return None, None
        logging.info("Данные успешно загружены.")

        start_preprocess_time = time.time()
        preprocessed_data = preprocess_data(data)
        if preprocessed_data is None or preprocessed_data.empty:
            return None, None
        logging.info("Данные успешно предобработаны.")
        preprocess_duration = time.time() - start_preprocess_time
        logging.info(f"Время предобработки данных: {preprocess_duration} секунд.")

        label = preprocessed_data.iloc[:, -1]

        start_model_time = time.time()

        # Обучение модели
        kwargs = {
            "presets": 'best_quality',
            "auto_stack": auto_stack,
            "num_cpus": 4,
            "num_bag_folds": num_bagging_folds  # Используем num_bag_folds вместо num_bagging_folds
        }
        predictor = TabularPredictor(label='symbol').fit(train_data=preprocessed_data, verbosity=3, **kwargs)
        logging.info("Модель успешно обучена.")

        model_duration = time.time() - start_model_time
        logging.info(f"Время обучения модели: {model_duration} секунд.")

        total_duration = time.time() - start_total_time
        logging.info(f"Общее время выполнения: {total_duration} секунд.")

        # Сохранение модели
        predictor.save("autogluon_model")
        
        return preprocessed_data, predictor

    except Exception as e:
        logging.error(f"Ошибка в процессе обучения модели: {str(e)}")
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

        
        # Получение новых данных для прогнозирования
        # Здесь нужно реализовать ваш код для загрузки новых данных

        # Предварительная обработка данных
        new_data = preprocess_data_for_prediction(new_data)
        if new_data is None:
            print("Ошибка при предварительной обработке данных для прогнозирования.")
            return

        # Прогнозирование
        predictions = make_predictions(new_data, predictor)
        if predictions is None:
            print("Ошибка при прогнозировании.")
            return

        # Визуализация результатов прогнозирования
        visualize_data_and_predictions(new_data, predictions)

    else:
        print("Во время обучения модели произошла ошибка. Проверьте журналы для получения дополнительной информации.")

if __name__ == "__main__":
    main()
