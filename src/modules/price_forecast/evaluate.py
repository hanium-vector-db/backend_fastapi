# evaluate.py
# -*- coding: utf-8 -*-
"""
모델 평가 및 메트릭 계산
"""
import math
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from modules.data_utils import sMAPE


def evaluate_model(model, loader, scaler, device):
    """
    모델 평가 함수

    Args:
        model: 학습된 모델
        loader: DataLoader
        scaler: StandardScaler
        device: torch.device

    Returns:
        tuple: (RMSE, R², sMAPE)
    """
    model.eval()
    Y_true_list = []
    Y_pred_list = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            # 예측
            pred = model(X)

            # 스케일링 해제
            pred_real = scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).ravel()
            y_real = scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1)).ravel()

            Y_true_list.append(y_real)
            Y_pred_list.append(pred_real)

    # 리스트가 비어있으면 NaN 반환
    if not Y_true_list:
        return np.nan, np.nan, np.nan

    # 전체 예측 결과 결합
    Y_true = np.concatenate(Y_true_list)
    Y_pred = np.concatenate(Y_pred_list)

    # 메트릭 계산
    rmse = math.sqrt(mean_squared_error(Y_true, Y_pred))
    r2 = r2_score(Y_true, Y_pred)
    smape = sMAPE(Y_true, Y_pred)

    return rmse, r2, smape
