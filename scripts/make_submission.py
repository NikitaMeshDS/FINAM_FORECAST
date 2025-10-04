import pandas as pd
import numpy as np
import os
import joblib
from datetime import timedelta

# --------------------------
# Параметры
# --------------------------
TARGET_DAYS = 20
LAGS = [1, 2, 3, 5, 10]
WINDOWS = [3, 5, 10]
TICKER_COL = "ticker"
SAVE_DIR = "./ticker_models"

# --------------------------
# Чтение данных
# --------------------------
candles = pd.read_csv("/Users/nikitamesh/FINAM_FORECAST/data/candles.csv", parse_dates=["begin"])
tickers = candles[TICKER_COL].unique()

# --------------------------
# Функция подготовки данных с фичами
# --------------------------
def prepare_features_for_future(df, lag_days=LAGS, windows=WINDOWS):
    df = df.sort_values(["ticker", "begin"]).reset_index(drop=True)
    all_features = []

    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        # Лаги
        for lag in lag_days:
            g[f"close_lag_{lag}"] = g["close"].shift(lag)
            g[f"volume_lag_{lag}"] = g["volume"].shift(lag)
        # Скользящие окна
        for window in windows:
            g[f"close_ma_{window}"] = g["close"].rolling(window).mean()
            g[f"close_std_{window}"] = g["close"].rolling(window).std()
            g[f"volume_ma_{window}"] = g["volume"].rolling(window).mean()
            g[f"volume_std_{window}"] = g["volume"].rolling(window).std()
        g["close_diff_1"] = g["close"].diff(1)
        g["close_diff_5"] = g["close"].diff(5)
        all_features.append(g)

    return pd.concat(all_features, axis=0).reset_index(drop=True)

# --------------------------
# Генерация будущих дат (календарные дни)
# --------------------------
future_data_list = []
last_dates = candles.groupby("ticker")["begin"].max().to_dict()

for ticker in tickers:
    last_date = last_dates[ticker]
    # Генерируем календарные дни (включая выходные)
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=TARGET_DAYS*2, freq='D')  # генерируем больше дней на случай лагов
    df = pd.DataFrame({
        "begin": dates,
        "ticker": ticker,
        "close": np.nan,
        "volume": np.nan
    })
    future_data_list.append(df)

future_data = pd.concat(future_data_list, axis=0).reset_index(drop=True)

# --------------------------
# Объединяем с историей для расчета лагов
# --------------------------
full_data = pd.concat([candles, future_data], axis=0).reset_index(drop=True)
full_data_prepared = prepare_features_for_future(full_data)

# Берем только будущие дни после последней даты
future_prepared = full_data_prepared[full_data_prepared["begin"] > candles["begin"].max()]

# --------------------------
# Предсказания
# --------------------------
returns_dict = {}

for ticker in tickers:
    print(f"Предсказываем для {ticker}...")
    model_path = os.path.join(SAVE_DIR, f"{ticker}_model.pkl")
    model = joblib.load(model_path)

    df_ticker = future_prepared[future_prepared["ticker"] == ticker].copy()
    feature_cols = [c for c in df_ticker.columns if c not in ["begin", "ticker", "close"]]

    # Используем только TARGET_DAYS первых дней для предсказаний
    df_ticker = df_ticker.head(TARGET_DAYS)
    y_pred = model.predict(df_ticker[feature_cols])
    df_ticker["pred_close"] = y_pred

    # Доходности от последнего реального дня с учетом выходных
    close_series = df_ticker["pred_close"].values
    dates_series = df_ticker["begin"].values
    
    # Получаем последнюю реальную цену для данного тикера
    last_real_close = candles[candles["ticker"] == ticker]["close"].iloc[-1]
    
    # Создаем массив доходностей для всех 20 календарных дней
    returns = [np.nan] * TARGET_DAYS  # Инициализируем все как NaN
    
    # Заполняем доходности только для будних дней
    for i in range(TARGET_DAYS):
        current_date = dates_series[i]
        weekday = pd.Timestamp(current_date).weekday()
        # Проверяем, является ли день выходным (суббота=5, воскресенье=6)
        if weekday < 5:  # 0-4 = понедельник-пятница
            # Доходность рассчитывается относительно последнего реального дня
            returns[i] = close_series[i] / last_real_close - 1
    
    returns_dict[ticker] = returns

# --------------------------
# Матрица доходностей тикеры × дни
# --------------------------
returns_matrix = pd.DataFrame(
    returns_dict,
    index=[f"p{i}" for i in range(1, TARGET_DAYS+1)]
).T
returns_matrix.index.name = "ticker"

returns_matrix.to_csv("/Users/nikitamesh/FINAM_FORECAST/data/submission.csv", na_rep='NaN')
print("Матрица доходностей сохранена в data/submission.csv")