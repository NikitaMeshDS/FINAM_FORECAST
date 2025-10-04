import pandas as pd
import os
from lightgbm import LGBMRegressor
import joblib

# --------------------------
# Параметры
# --------------------------
LAGS = [1, 2, 3, 5, 10]
WINDOWS = [3, 5, 10]
TICKER_COL = "ticker"
SAVE_DIR = "./ticker_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------
# Чтение данных
# --------------------------
#отредачить
candles = pd.read_csv("/Users/nikitamesh/FINAM_FORECAST/data/candles.csv", parse_dates=["begin"])

# --------------------------
# Подготовка данных с фичами
# --------------------------
def prepare_data(df, lags=LAGS, windows=WINDOWS):
    df = df.sort_values([TICKER_COL, "begin"]).reset_index(drop=True)
    all_features = []

    for ticker, group in df.groupby(TICKER_COL):
        g = group.copy()
        for lag in lags:
            g[f"close_lag_{lag}"] = g["close"].shift(lag)
            g[f"volume_lag_{lag}"] = g["volume"].shift(lag)
        for window in windows:
            g[f"close_ma_{window}"] = g["close"].rolling(window).mean()
            g[f"close_std_{window}"] = g["close"].rolling(window).std()
            g[f"volume_ma_{window}"] = g["volume"].rolling(window).mean()
            g[f"volume_std_{window}"] = g["volume"].rolling(window).std()
        g["close_diff_1"] = g["close"].diff(1)
        g["close_diff_5"] = g["close"].diff(5)
        g = g.dropna().reset_index(drop=True)
        all_features.append(g)
    return pd.concat(all_features, axis=0).reset_index(drop=True)

train_data = prepare_data(candles)
tickers = candles[TICKER_COL].unique()

# --------------------------
# Обучение моделей
# --------------------------
def train_models(train_data, tickers, save_dir=SAVE_DIR):
    models = {}
    for ticker in tickers:
        print(f"Обучаем модель для {ticker}...")
        data = train_data[train_data[TICKER_COL] == ticker]
        feature_cols = [c for c in data.columns if c not in ["begin", "ticker", "close"]]
        X_train = data[feature_cols]
        y_train = data["close"]
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )
        model.fit(X_train, y_train)
        models[ticker] = model
        # Сохраняем модель через joblib
        joblib.dump(model, os.path.join(save_dir, f"{ticker}_model.pkl"))
    return models

models = train_models(train_data, tickers)
print("Обучение завершено. Модели сохранены.")