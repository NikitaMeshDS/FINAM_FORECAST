import pandas as pd
import os
from lightgbm import LGBMRegressor
import joblib
from config import TARGET_DAYS, LAGS, WINDOWS, TICKER_COL, SAVE_DIR, DATA_DIR

# Подготовка данных с фичами
def prepare_data(df, lags=LAGS, windows=WINDOWS):
    """Подготавливает данные с техническими индикаторами и новостными признаками"""
    df = df.sort_values([TICKER_COL, "begin"]).reset_index(drop=True)
    all_features = []

    for ticker, group in df.groupby(TICKER_COL):
        g = group.copy()
        
        # Технические индикаторы
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
        
        # Дополнительные технические признаки
        g["high_low_ratio"] = g["high"] / g["low"]
        g["open_close_ratio"] = g["open"] / g["close"]
        g["volume_price_ratio"] = g["volume"] / g["close"]
        
        # Удаляем строки с NaN (первые строки из-за лагов)
        g = g.dropna().reset_index(drop=True)
        all_features.append(g)
    
    return pd.concat(all_features, axis=0).reset_index(drop=True)

# Обучение моделей
def train_models(train_data, tickers, save_dir=SAVE_DIR):
    """Обучает модели для каждого тикера"""
    models = {}
    
    for ticker in tickers:
        print(f"Обучаем модель для {ticker}...")
        data = train_data[train_data[TICKER_COL] == ticker].copy()
        
        # Определяем признаки (исключаем служебные колонки)
        exclude_cols = ["begin", "ticker", "close", "begin_date_only", "open", "high", "low", "volume"]
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        X_train = data[feature_cols]
        y_train = data["close"]
        
        print(f"  Количество признаков: {len(feature_cols)}")
        print(f"  Размер обучающей выборки: {len(X_train)}")
        
        # Настройки модели LightGBM
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        models[ticker] = model
        
        # Сохраняем модель
        model_path = os.path.join(save_dir, f"{ticker}_model.pkl")
        joblib.dump(model, model_path)
        
        # Сохраняем список признаков для использования в prediction
        feature_info = {
            'feature_cols': feature_cols,
            'ticker': ticker
        }
        joblib.dump(feature_info, os.path.join(save_dir, f"{ticker}_features.pkl"))
        
        print(f"  Модель сохранена: {model_path}")
    
    return models

if __name__ == "__main__":
    # Чтение объединенного датасета
    combined_data_path = os.path.join(DATA_DIR, "combined_dataset.csv")
    print(f"Загружаем данные из: {combined_data_path}")
    
    combined_data = pd.read_csv(combined_data_path, parse_dates=["begin"])
    print(f"Загружено {len(combined_data)} записей")
    print(f"Количество тикеров: {combined_data['ticker'].nunique()}")
    
    # Подготавливаем данные с признаками
    train_data = prepare_data(combined_data)
    print(f"После подготовки признаков: {len(train_data)} записей")
    
    # Получаем список тикеров
    tickers = combined_data[TICKER_COL].unique()
    print(f"Тикеры для обучения: {tickers}")
    
    # Обучаем модели
    models = train_models(train_data, tickers)
    print("Обучение завершено. Модели сохранены.")