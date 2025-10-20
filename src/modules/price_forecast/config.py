# config.py
# -*- coding: utf-8 -*-
"""
설정 및 상수 정의
"""
import os
import random
import numpy as np
import torch

# ===== 경로 설정 =====
BASE_PATH = "/home/ubuntu_euphoria/Desktop/hanium/ai"
DIR_DATA = os.path.join(BASE_PATH, "data")  # CSV 데이터 디렉토리
DIR_MODELS = os.path.join(BASE_PATH, "models")
DIR_PLOTS = os.path.join(BASE_PATH, "plots")
DIR_FUTURE = os.path.join(BASE_PATH, "future")
DIR_METRICS = os.path.join(BASE_PATH, "metrics")
DIR_TENSORBOARD = os.path.join(BASE_PATH, "runs")  # TensorBoard 로그 디렉토리

# 디렉토리 생성
for d in [DIR_MODELS, DIR_PLOTS, DIR_FUTURE, DIR_METRICS, DIR_TENSORBOARD]:
    os.makedirs(d, exist_ok=True)

# ===== 데이터 파라미터 =====
LOOKBACK_MAX = 15  # 최대 윈도우 길이
HORIZON_MAX = 3    # 최대 예측 일수
SPLIT_TRY = ("7:2:1", "8:1:1", "9:0.5:0.5", "9.5:0.3:0.2")  # 분할 시도 비율
INTERP_MAX_GAP = 21  # 최대 보간 공백 (일)

# ===== 학습 파라미터 =====
BATCH_SIZE = 256
EPOCHS = 2000
LR = 1e-4
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.3
PATIENCE = 10  # Early stopping patience
SCHEDULER_PATIENCE = 3  # LR scheduler patience

# ===== 예측 파라미터 =====
MAX_FUTURE_DAYS = 90
FUTURE_START_DATE = "2025-08-01"  # "auto"면 마지막 관측일+1부터
FUTURE_DAYS = 60

# ===== 시드 설정 =====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ===== 디바이스 설정 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 과일 목록 =====
FILES = [
    "망고 1과 가격정보.csv",
    "오렌지(미국산 네이블) 10알 가격정보.csv",
    "수입 파인애플 1과.csv",
    "감귤(노지) 10알 가격정보.csv",
    "바나나 100g 가격정보.csv",
    "배(신고배) 10알 가격정보.csv",
    "샤인머스캣 2kg 가격정보.csv",
    "복숭아(백도) 10알 가격정보.csv",
    "방울토마토 1kg 가격정보.csv",
    "딸기 100g 가격정보.csv",
]

# ===== 영어 과일 이름 매핑 =====
FRUIT_NAME_EN = {
    "망고 1과 가격정보": "Mango_1pc",
    "오렌지(미국산 네이블) 10알 가격정보": "Orange_US_Navel_10pc",
    "수입 파인애플 1과": "Pineapple_Imported_1pc",
    "감귤(노지) 10알 가격정보": "Tangerine_10pc",
    "바나나 100g 가격정보": "Banana_100g",
    "배(신고배) 10알 가격정보": "Pear_Shingo_10pc",
    "샤인머스캣 2kg 가격정보": "Shine_Muscat_2kg",
    "복숭아(백도) 10알 가격정보": "Peach_Baekdo_10pc",
    "방울토마토 1kg 가격정보": "Cherry_Tomato_1kg",
    "딸기 100g 가격정보": "Strawberry_100g",
}

# ===== 데이터 범위 =====
DATA_START_DATE = "2021-01-01"
DATA_END_DATE = "2025-07-31"
