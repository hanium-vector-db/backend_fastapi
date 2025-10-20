# train.py
# -*- coding: utf-8 -*-
"""
모델 학습 로직
"""
import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modules.config import EPOCHS, LR, PATIENCE, SCHEDULER_PATIENCE, DIR_TENSORBOARD
from modules.evaluate import evaluate_model


def train_model(
    model,
    train_loader,
    val_loader,
    scaler,
    device,
    fruit_name,
    save_dir
):
    """
    모델 학습 함수

    Args:
        model: 학습할 모델
        train_loader: 학습 DataLoader
        val_loader: 검증 DataLoader
        scaler: StandardScaler
        device: torch.device
        fruit_name: 과일 이름 (저장용)
        save_dir: 모델 저장 디렉토리

    Returns:
        str: 저장된 best 모델 경로
    """
    # TensorBoard Writer 초기화
    writer = SummaryWriter(log_dir=os.path.join(DIR_TENSORBOARD, fruit_name))

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=SCHEDULER_PATIENCE
    )

    # Early Stopping
    best_rmse = float('inf')
    patience_counter = 0

    # 모델 저장 경로
    best_pt = os.path.join(save_dir, f"{fruit_name}_best.model.pt")
    best_pth = os.path.join(save_dir, f"{fruit_name}_Best.model.pth")

    # 학습 히스토리 저장
    history = {
        'epoch': [],
        'train_loss': [],
        'val_rmse': [],
        'val_r2': [],
        'val_smape': [],
        'lr': []
    }

    # 학습 루프
    for epoch in tqdm(range(1, EPOCHS + 1), desc=f"[{fruit_name}] Training", leave=False):
        # Training Phase
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            # Forward
            pred = model(X)
            loss = criterion(pred, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        val_rmse, val_r2, val_smape = evaluate_model(model, val_loader, scaler, device)

        # TensorBoard 로깅
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Metrics/val_RMSE', val_rmse, epoch)
        writer.add_scalar('Metrics/val_R2', val_r2, epoch)
        writer.add_scalar('Metrics/val_sMAPE', val_smape, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 히스토리 저장
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['val_smape'].append(val_smape)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Learning Rate Scheduling
        scheduler.step(val_rmse)

        # 로그 출력
        tqdm.write(
            f"[{fruit_name}][Epoch {epoch:04d}] "
            f"Train Loss={avg_train_loss:.6f} | "
            f"Val RMSE={val_rmse:.4f} | R²={val_r2:.4f} | sMAPE={val_smape:.2f}%"
        )

        # Best 모델 저장
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_pt)
            torch.save(model.state_dict(), best_pth)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= PATIENCE:
            tqdm.write(f"[{fruit_name}] Early stopping at epoch {epoch}")
            break

    # TensorBoard Writer 닫기
    writer.close()

    # Best 모델 경로와 히스토리 반환
    return (best_pt if os.path.exists(best_pt) else best_pth), history
