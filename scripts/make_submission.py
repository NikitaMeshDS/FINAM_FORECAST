import pandas as pd
import numpy as np
import os
import joblib
from datetime import timedelta
from config import TARGET_DAYS, LAGS, WINDOWS, TICKER_COL, SAVE_DIR, DATA_PATH, SUBMISSION_PATH

# Функция подготовки данных с фичами
def prepare_features_for_future(df, lag_days=LAGS, windows=WINDOWS):
    """Подготавливает признаки для будущих предсказаний"""
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

# Функция генерации будущих дат
def generate_future_dates(candles, target_days=TARGET_DAYS):
    """Генерирует календарные дни для предсказаний"""
    future_data_list = []
    last_dates = candles.groupby("ticker")["begin"].max().to_dict()
    tickers = candles[TICKER_COL].unique()

    for ticker in tickers:
        last_date = last_dates[ticker]
        # Генерируем календарные дни (включая выходные)
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=target_days*2, freq='D')
        df = pd.DataFrame({
            "begin": dates,
            "ticker": ticker,
            "close": np.nan,
            "volume": np.nan
        })
        future_data_list.append(df)

    return pd.concat(future_data_list, axis=0).reset_index(drop=True)

# Функция предсказаний
def make_predictions(future_prepared, candles, save_dir=SAVE_DIR, target_days=TARGET_DAYS):
    """Делает предсказания для всех тикеров"""
    returns_dict = {}
    tickers = candles[TICKER_COL].unique()

    for ticker in tickers:
        print(f"Предсказываем для {ticker}...")
        model_path = os.path.join(save_dir, f"{ticker}_model.pkl")
        model = joblib.load(model_path)

        df_ticker = future_prepared[future_prepared["ticker"] == ticker].copy()
        feature_cols = [c for c in df_ticker.columns if c not in ["begin", "ticker", "close"]]

        # Используем только TARGET_DAYS первых дней для предсказаний
        df_ticker = df_ticker.head(target_days)
        y_pred = model.predict(df_ticker[feature_cols])
        df_ticker["pred_close"] = y_pred

        # Доходности от последнего реального дня с учетом выходных
        close_series = df_ticker["pred_close"].values
        dates_series = df_ticker["begin"].values
        
        # Получаем последнюю реальную цену для данного тикера
        last_real_close = candles[candles["ticker"] == ticker]["close"].iloc[-1]
        
        # Создаем массив доходностей для всех календарных дней
        returns = [np.nan] * target_days
        
        # Заполняем доходности только для будних дней
        for i in range(target_days):
            current_date = dates_series[i]
            weekday = pd.Timestamp(current_date).weekday()
            # Проверяем, является ли день выходным (суббота=5, воскресенье=6)
            if weekday < 5:
                returns[i] = close_series[i] / last_real_close - 1
        
        returns_dict[ticker] = returns
    
    return returns_dict

# Функция создания submission файла
def make_submission(returns_dict, target_days=TARGET_DAYS, output_path=SUBMISSION_PATH):
    """Создает матрицу доходностей и сохраняет в CSV"""
    returns_matrix = pd.DataFrame(
        returns_dict,
        index=[f"p{i}" for i in range(1, target_days+1)]
    ).T
    returns_matrix.index.name = "ticker"

    returns_matrix.to_csv(output_path, na_rep='NaN')
    print(f"Матрица доходностей сохранена в {output_path}")


# Основная функция
def main():
    """Основная функция для создания submission файла"""
    # Чтение данных
    candles = pd.read_csv(DATA_PATH, parse_dates=["begin"])
    
    # Генерация будущих дат
    future_data = generate_future_dates(candles)
    
    # Объединяем с историей для расчета лагов
    full_data = pd.concat([candles, future_data], axis=0).reset_index(drop=True)
    full_data_prepared = prepare_features_for_future(full_data)
    
    # Берем только будущие дни после последней даты
    future_prepared = full_data_prepared[full_data_prepared["begin"] > candles["begin"].max()]
    
    # Делаем предсказания
    returns_dict = make_predictions(future_prepared, candles)
    
    # Создаем submission файл
    make_submission(returns_dict)

if __name__ == "__main__":
    main()