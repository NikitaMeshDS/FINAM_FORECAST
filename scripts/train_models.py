import pandas as pd
import os
from lightgbm import LGBMRegressor
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
        
        # Заполняем NaN пропущенные значения
        g = g.ffill().bfill()
        all_features.append(g)
    
    return pd.concat(all_features, axis=0).reset_index(drop=True)

# Обучение моделей
def train_models(train_data, tickers, save_dir=SAVE_DIR):
    """Обучает модели для каждого тикера с перебором параметров"""
    models = {}
    
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [6, 8],
        'min_child_samples': [20, 30]
    }
    
    for ticker in tickers:
        print(f"Обучаем модель для {ticker}...")
        data = train_data[train_data[TICKER_COL] == ticker].copy()
        
        # Определяем признаки (исключаем служебные колонки)
        exclude_cols = ["begin", "ticker", "close", "begin_date_only", "open", "high", "low", "volume"]
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        
        X_train = data[feature_cols]
        y_train = data["close"]
        
        # Базовые настройки модели LightGBM
        base_model = LGBMRegressor(
            random_state=42,
            verbose=-1
        )
        
        # Используем TimeSeriesSplit для временных рядов
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Получаем лучшую модель
        best_model = grid_search.best_estimator_
        models[ticker] = best_model
        
        model_path = os.path.join(save_dir, f"{ticker}_model.pkl")
        joblib.dump(best_model, model_path)
        
        # Сохраняем список признаков для использования в prediction
        feature_info = {
            'feature_cols': feature_cols,
            'ticker': ticker,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        joblib.dump(feature_info, os.path.join(save_dir, f"{ticker}_features.pkl"))
    
    return models

if __name__ == "__main__":а
    combined_data_path = os.path.join(DATA_DIR, "combined_dataset.csv")
    
    combined_data = pd.read_csv(combined_data_path, parse_dates=["begin"])
    
    # Подготавливаем данные с признаками
    train_data = prepare_data(combined_data)
    
    # Получаем список тикеров
    tickers = combined_data[TICKER_COL].unique()
    
    models = train_models(train_data, tickers)