import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from config import LAGS, WINDOWS, TICKER_COL, SAVE_DIR, COMBINED_DATASET_PATH

def prepare_data(df, lags=LAGS, windows=WINDOWS):
    """Подготавливает данные с техническими индикаторами и новостными признаками"""
    df = df.sort_values([TICKER_COL, "begin"]).reset_index(drop=True)
    all_features = []

    for _, group in df.groupby(TICKER_COL):
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
        
        #Дополнительные технические признаки
        g["high_low_ratio"] = g["high"] / g["low"]
        g["open_close_ratio"] = g["open"] / g["close"]
        g["volume_price_ratio"] = g["volume"] / g["close"]
        
        g = g.ffill().bfill()
        all_features.append(g)
    
    return pd.concat(all_features, axis=0).reset_index(drop=True)

def mean_absolute_percentage_error(y_true, y_pred):
    """Вычисляет среднюю абсолютную процентную ошибку (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_metrics(y_true, y_pred):
    """Вычисляет все метрики для оценки модели"""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Max_Error': np.max(np.abs(y_true - y_pred)),
        'Mean_Error': np.mean(y_pred - y_true)
    }
    return metrics

def evaluate_model_on_data(model, X, feature_cols):
    """Оценивает модель на данных"""
    # Проверяем наличие всех признаков
    missing_features = [col for col in feature_cols if col not in X.columns]
    if missing_features:
        for col in missing_features:
            X[col] = 0.0
    
    # Заполняем NaN значения
    X[feature_cols] = X[feature_cols].fillna(0)
    
    y_pred = model.predict(X[feature_cols])
    return y_pred

def evaluate_models_cv(train_data, tickers, save_dir=SAVE_DIR, n_splits=3):
    """Оценивает модели с использованием кросс-валидации"""
    results = {}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for ticker in tickers:
        print(f"\nОцениваем модель для {ticker}...")
        
        model_path = os.path.join(save_dir, f"{ticker}_model.pkl")
        features_path = os.path.join(save_dir, f"{ticker}_features.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            print(f"  Модель или файл признаков не найдены для {ticker}, пропускаем")
            continue
        
        model = joblib.load(model_path)
        feature_info = joblib.load(features_path)
        feature_cols = feature_info['feature_cols']
        
        data = train_data[train_data[TICKER_COL] == ticker].copy()
        
        exclude_cols = ["begin", "ticker", "close", "begin_date_only", "open", "high", "low", "volume",
                        "open_close_ratio", "volume_price_ratio"]
        X = data[[c for c in data.columns if c not in exclude_cols]]
        y = data["close"]
        
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < n_splits + 1:
            print(f"  Недостаточно данных для кросс-валидации для {ticker}")
            continue
        
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Обучаем модель на фолде
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train_fold[feature_cols], y_train_fold)
            
            # Предсказания
            y_pred_fold = model_fold.predict(X_val_fold[feature_cols])
            
            # Метрики
            fold_metrics = calculate_metrics(y_val_fold, y_pred_fold)
            fold_metrics['fold'] = fold + 1
            cv_metrics.append(fold_metrics)
            
            print(f"  Fold {fold + 1}: MAE={fold_metrics['MAE']:.4f}, RMSE={fold_metrics['RMSE']:.4f}, MAPE={fold_metrics['MAPE']:.2f}%")
        
        # Средние метрики по всем фолдам
        avg_metrics = {}
        for key in ['MAE', 'RMSE', 'MAPE', 'R2', 'Max_Error', 'Mean_Error']:
            avg_metrics[key] = np.mean([m[key] for m in cv_metrics])
            avg_metrics[f'{key}_std'] = np.std([m[key] for m in cv_metrics])
        
        results[ticker] = {
            'cv_metrics': cv_metrics,
            'avg_metrics': avg_metrics
        }
        
        print(f"  Средние метрики: MAE={avg_metrics['MAE']:.4f}±{avg_metrics['MAE_std']:.4f}, "
              f"RMSE={avg_metrics['RMSE']:.4f}±{avg_metrics['RMSE_std']:.4f}, "
              f"MAPE={avg_metrics['MAPE']:.2f}%±{avg_metrics['MAPE_std']:.2f}%")
    
    return results

def evaluate_models_simple(train_data, tickers, save_dir=SAVE_DIR, test_size=0.2):
    """Оценивает модели на простом train/test split"""
    results = {}
    
    for ticker in tickers:
        print(f"\nОцениваем модель для {ticker}...")
        
        model_path = os.path.join(save_dir, f"{ticker}_model.pkl")
        features_path = os.path.join(save_dir, f"{ticker}_features.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            print(f"  Модель или файл признаков не найдены для {ticker}, пропускаем")
            continue
        
        model = joblib.load(model_path)
        feature_info = joblib.load(features_path)
        feature_cols = feature_info['feature_cols']
        
        data = train_data[train_data[TICKER_COL] == ticker].copy().sort_values("begin")
        
        exclude_cols = ["begin", "ticker", "close", "begin_date_only", "open", "high", "low", "volume",
                        "open_close_ratio", "volume_price_ratio"]  # Исключаем признаки с утечкой данных
        X = data[[c for c in data.columns if c not in exclude_cols]]
        y = data["close"]
        
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            print(f"  Недостаточно данных для оценки для {ticker}")
            continue
        
        # Разделение на train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Предсказания на тестовой выборке
        y_pred = evaluate_model_on_data(model, X_test, feature_cols)
        
        # Метрики
        metrics = calculate_metrics(y_test, y_pred)
        results[ticker] = metrics
        
        print(f"  MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
              f"MAPE={metrics['MAPE']:.2f}%, R²={metrics['R2']:.4f}")
    
    return results

def print_summary(results):
    """Выводит сводку по всем метрикам"""
    if not results:
        print("\nНет результатов для вывода")
        return
    
    print("\n" + "="*80)
    print("СВОДКА ПО МЕТРИКАМ")
    print("="*80)
    
    # Определяем тип результатов (CV или простой)
    first_ticker = list(results.keys())[0]
    if 'avg_metrics' in results[first_ticker]:
        # CV результаты
        print(f"\n{'Тикер':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<12} {'R²':<10}")
        print("-" * 80)
        
        for ticker, result in results.items():
            m = result['avg_metrics']
            print(f"{ticker:<10} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} "
                  f"{m['MAPE']:<12.2f}% {m['R2']:<10.4f}")
        
        # Общие средние
        all_mae = [r['avg_metrics']['MAE'] for r in results.values()]
        all_rmse = [r['avg_metrics']['RMSE'] for r in results.values()]
        all_mape = [r['avg_metrics']['MAPE'] for r in results.values()]
        all_r2 = [r['avg_metrics']['R2'] for r in results.values()]
        
    else:
        # Простые результаты
        print(f"\n{'Тикер':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<12} {'R²':<10}")
        print("-" * 80)
        
        for ticker, metrics in results.items():
            print(f"{ticker:<10} {metrics['MAE']:<12.4f} {metrics['RMSE']:<12.4f} "
                  f"{metrics['MAPE']:<12.2f}% {metrics['R2']:<10.4f}")
        
        all_mae = [r['MAE'] for r in results.values()]
        all_rmse = [r['RMSE'] for r in results.values()]
        all_mape = [r['MAPE'] for r in results.values()]
        all_r2 = [r['R2'] for r in results.values()]
    
    print("-" * 80)
    print(f"{'СРЕДНЕЕ':<10} {np.mean(all_mae):<12.4f} {np.mean(all_rmse):<12.4f} "
          f"{np.mean(all_mape):<12.2f}% {np.mean(all_r2):<10.4f}")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка метрик моделей прогнозирования')
    parser.add_argument('--cv', action='store_true', help='Использовать кросс-валидацию')
    parser.add_argument('--n_splits', type=int, default=3, help='Количество фолдов для CV')
    parser.add_argument('--test_size', type=float, default=0.2, help='Размер тестовой выборки (для простой оценки)')
    
    args = parser.parse_args()
    
    print("Загрузка данных...")
    combined_data = pd.read_csv(COMBINED_DATASET_PATH, parse_dates=["begin"])
    
    print("Подготовка признаков...")
    train_data = prepare_data(combined_data)
    
    tickers = combined_data[TICKER_COL].unique()
    print(f"Найдено тикеров: {len(tickers)}")
    
    if args.cv:
        print("\nОценка с использованием кросс-валидации...")
        results = evaluate_models_cv(train_data, tickers, n_splits=args.n_splits)
    else:
        print("\nОценка на тестовой выборке...")
        results = evaluate_models_simple(train_data, tickers, test_size=args.test_size)
    
    print_summary(results)

