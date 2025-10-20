# main.py
# -*- coding: utf-8 -*-
"""
과일 가격 예측 모델 학습 및 평가 메인 스크립트
"""
import os
import pickle
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# 한글 폰트 설정
if platform.system() == "Windows":
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == "Darwin":
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# 모듈 임포트
from modules.config import (
    BASE_PATH, DIR_DATA, DIR_MODELS, DIR_PLOTS, DIR_FUTURE, DIR_METRICS,
    FILES, FRUIT_NAME_EN, BATCH_SIZE, HIDDEN, LAYERS, DROPOUT, DEVICE,
    SPLIT_TRY, INTERP_MAX_GAP, DATA_START_DATE, DATA_END_DATE,
    FUTURE_START_DATE, FUTURE_DAYS
)
from modules.data_utils import (
    safe_read_csv, seasonal_resample, longest_valid_block,
    split_block_lenient, adapt_window_for_block, add_time_features
)
from modules.model import LSTMForecaster
from modules.dataset import SingleFruitDataset
from modules.train import train_model
from modules.evaluate import evaluate_model
from modules.predict import predict_future


def build_loader(series, split_part, lookback, horizon, scaler, split_name):
    """
    DataLoader 생성 함수
    """
    dataset = SingleFruitDataset(series, split_part.index, lookback, horizon, scaler)
    shuffle = (split_name == 'train')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    return loader, len(dataset)


def visualize_test_prediction(model, series, test_part, lookback, horizon, scaler, fruit_name, device):
    """
    테스트 구간 예측 시각화 (전체 구간)
    """
    import torch
    import numpy as np

    # 데이터 준비
    price = pd.DataFrame({'price': series})
    feats = add_time_features(price.index)
    frame = pd.concat([price, feats], axis=1)
    frame['price_scaled'] = scaler.transform(frame[['price']].values)

    X_cols = ['price_scaled', 'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow', 'month_norm', 'year_norm']

    # 전체 데이터에서 테스트 구간까지 포함
    test_start_idx = series.index.get_loc(test_part.index[0])
    test_end_idx = series.index.get_loc(test_part.index[-1])

    # 예측을 위해 lookback 포함한 구간 준비
    start_idx = max(0, test_start_idx - lookback)
    end_idx = min(len(series), test_end_idx + 1)

    subset = frame.iloc[start_idx:end_idx]
    arr = subset[X_cols].values

    # 슬라이딩 윈도우로 테스트 구간 전체 예측
    predictions = []
    pred_dates = []
    actuals = []
    actual_dates = []

    model.eval()

    # 테스트 구간의 각 포인트에 대해 예측
    for i in range(len(subset) - lookback):
        window = arr[i:i + lookback]

        # NaN 체크
        if np.isnan(window).any():
            continue

        # 예측 날짜가 테스트 구간에 있는지 확인
        pred_date = subset.index[i + lookback]
        if pred_date not in test_part.index:
            continue

        # 예측 수행
        X_in = torch.tensor(window.astype('float32')).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(X_in).cpu().numpy().ravel()

        # 첫 번째 예측값만 사용 (1-step ahead)
        pred_price = scaler.inverse_transform(pred_scaled[:1].reshape(-1, 1)).ravel()[0]
        actual_price = subset['price'].iloc[i + lookback]

        predictions.append(pred_price)
        pred_dates.append(pred_date)
        actuals.append(actual_price)
        actual_dates.append(pred_date)

    if len(predictions) == 0:
        tqdm.write(f"[WARN] {fruit_name}: No valid test predictions")
        return

    # 그래프 생성
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 상단: 전체 비교
    axes[0].plot(actual_dates, actuals, label='Actual', marker='o', linewidth=2, markersize=4)
    axes[0].plot(pred_dates, predictions, label='Predicted', marker='x', linewidth=2, markersize=4, alpha=0.8)
    axes[0].set_title(f"{fruit_name} | Test Set Predictions (n={len(predictions)})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=11)
    axes[0].set_ylabel('Price', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # 하단: 오차 분석
    errors = np.array(predictions) - np.array(actuals)
    axes[1].bar(actual_dates, errors, width=1.0, alpha=0.7, color=['red' if e > 0 else 'blue' for e in errors])
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title(f"Prediction Errors (Red=Overestimate, Blue=Underestimate)", fontsize=12)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Error (Predicted - Actual)', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)

    # 통계 정보 추가
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    fig.text(0.99, 0.01, f'MAE: {mae:.2f} | RMSE: {rmse:.2f}',
             ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, f"{fruit_name}_test_prediction.png"), dpi=150)
    plt.close()


def main():
    """
    메인 실행 함수
    """
    print("=" * 80)
    print("과일 가격 예측 모델 학습 시작")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"학습 과일 개수: {len(FILES)}")
    print("=" * 80)

    metrics_rows = []

    for file_name in tqdm(FILES, desc="과일별 학습 진행"):
        fruit_name_kr = os.path.splitext(file_name)[0]
        fruit_name = FRUIT_NAME_EN.get(fruit_name_kr, fruit_name_kr)  # 영어 이름으로 변환
        csv_path = os.path.join(DIR_DATA, file_name)

        # 1. 데이터 로딩
        if not os.path.exists(csv_path):
            tqdm.write(f"[SKIP] {fruit_name}: CSV 파일 없음 ({csv_path})")
            continue

        df = safe_read_csv(csv_path)
        date_col, price_col = df.columns[0], df.columns[1]

        # 2. 리샘플링 및 필터링
        series = seasonal_resample(df, date_col, price_col, INTERP_MAX_GAP)
        series = series[(series.index >= DATA_START_DATE) & (series.index <= DATA_END_DATE)]

        # 3. 가장 긴 연속 블록 추출
        block = longest_valid_block(series)
        if block is None:
            tqdm.write(f"[SKIP] {fruit_name}: 연속 블록 없음")
            continue

        # 4. 윈도우 크기 자동 조정
        lookback, horizon = adapt_window_for_block(len(block))
        min_len = lookback + horizon + 5

        # 5. Train/Val/Test 분할
        train_part, val_part, test_part = split_block_lenient(block, min_len, SPLIT_TRY)
        if train_part is None:
            tqdm.write(f"[SKIP] {fruit_name}: 블록 길이 부족 (최소 {min_len}일 필요)")
            continue

        tqdm.write(
            f"[{fruit_name}] Block={len(block)} | "
            f"Train={len(train_part)} Val={len(val_part)} Test={len(test_part)} | "
            f"LB={lookback} HZ={horizon}"
        )

        # 6. 스케일러 fit
        scaler = StandardScaler().fit(train_part.values.reshape(-1, 1))
        scaler_path = os.path.join(DIR_MODELS, f"{fruit_name}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # 7. DataLoader 생성
        train_loader, n_train = build_loader(series, train_part, lookback, horizon, scaler, 'train')
        val_loader, n_val = build_loader(series, val_part, lookback, horizon, scaler, 'val')
        test_loader, n_test = build_loader(series, test_part, lookback, horizon, scaler, 'test')

        if n_train == 0 or n_val == 0 or n_test == 0:
            tqdm.write(f"[SKIP] {fruit_name}: 윈도우 생성 실패")
            continue

        # 8. 모델 생성
        model = LSTMForecaster(
            input_dim=7,
            hidden=HIDDEN,
            layers=LAYERS,
            dropout=DROPOUT,
            horizon=horizon
        ).to(DEVICE)

        # 9. 학습
        best_model_path, history = train_model(
            model, train_loader, val_loader, scaler, DEVICE, fruit_name, DIR_MODELS
        )

        # 10. 최종 평가
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        final_val = evaluate_model(model, val_loader, scaler, DEVICE)
        final_test = evaluate_model(model, test_loader, scaler, DEVICE)

        print(f"\n[{fruit_name}] Final Val : RMSE={final_val[0]:.4f} | R²={final_val[1]:.4f} | sMAPE={final_val[2]:.2f}%")
        print(f"[{fruit_name}] Final Test: RMSE={final_test[0]:.4f} | R²={final_test[1]:.4f} | sMAPE={final_test[2]:.2f}%\n")

        # 11. 테스트 예측 시각화
        visualize_test_prediction(model, series, test_part, lookback, horizon, scaler, fruit_name, DEVICE)

        # 11-1. 에폭별 학습 곡선 저장
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{fruit_name} Training History', fontsize=16)

        # Train Loss
        axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Train Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Validation RMSE
        axes[0, 1].plot(history['epoch'], history['val_rmse'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation RMSE')
        axes[0, 1].set_title('Validation RMSE')
        axes[0, 1].grid(True, alpha=0.3)

        # Validation R²
        axes[1, 0].plot(history['epoch'], history['val_r2'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation R²')
        axes[1, 0].set_title('Validation R²')
        axes[1, 0].grid(True, alpha=0.3)

        # Validation sMAPE
        axes[1, 1].plot(history['epoch'], history['val_smape'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation sMAPE (%)')
        axes[1, 1].set_title('Validation sMAPE')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(DIR_PLOTS, f"{fruit_name}_training_history.png"), dpi=150)
        plt.close()

        # 12. 미래 예측
        try:
            future_df = predict_future(
                model, series, scaler, lookback, FUTURE_DAYS, FUTURE_START_DATE, DEVICE
            )
            future_csv = os.path.join(DIR_FUTURE, f"{fruit_name}_future_from_{FUTURE_START_DATE}_{FUTURE_DAYS}d.csv")
            future_df.to_csv(future_csv, index=False)

            # Future prediction graph
            plt.figure(figsize=(12, 5))
            plt.plot(future_df['date'], future_df['pred_price'], marker='o', linewidth=2)
            plt.title(f"{fruit_name} | {FUTURE_DAYS}-day Forecast from {FUTURE_START_DATE}")
            plt.xlabel('Date')
            plt.ylabel('Predicted Price')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            future_png = os.path.join(DIR_FUTURE, f"{fruit_name}_future_from_{FUTURE_START_DATE}_{FUTURE_DAYS}d.png")
            plt.savefig(future_png, dpi=150)
            plt.close()
        except Exception as e:
            tqdm.write(f"[WARN] {fruit_name} 미래 예측 실패: {e}")

        # 13. 메트릭 저장
        metrics_rows.append({
            "fruit": fruit_name,  # 영어 이름으로 저장
            "val_RMSE": final_val[0], "val_R2": final_val[1], "val_sMAPE": final_val[2],
            "test_RMSE": final_test[0], "test_R2": final_test[1], "test_sMAPE": final_test[2],
            "best_model": best_model_path, "scaler": scaler_path
        })

    # 14. 메트릭 CSV 저장
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = os.path.join(DIR_METRICS, "per_fruit_metrics.csv")
        try:
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"\n[완료] 메트릭 저장: {metrics_csv}")
        except PermissionError:
            alt_path = os.path.join(DIR_METRICS, f"per_fruit_metrics_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")
            metrics_df.to_csv(alt_path, index=False)
            print(f"\n[WARN] 파일 잠금 → 대체 경로 저장: {alt_path}")
    else:
        print("\n[알림] 저장할 메트릭이 없습니다.")

    print("\n" + "=" * 80)
    print("모든 학습 완료!")
    print("=" * 80)


if __name__ == "__main__":
    import torch
    main()
