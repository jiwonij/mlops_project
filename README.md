# ⚡ Building Energy Forecasting MLOps Pipeline

시계열 데이터를 활용한 건물 전력 소비량 예측 모델을 구축하고,  
REST API 기반 자동 재학습(MLOps) 파이프라인을 구현한 프로젝트입니다.

---

## 🚀 프로젝트 개요

본 프로젝트는 다음과 같은 흐름을 자동화하는 것을 목표로 합니다.

1. CSV 데이터 업로드
2. 시계열 예측 수행 (LSTM)
3. 예측 성능 평가 (RMSE, MAE 등)
4. 성능 기준 초과 시 자동 재학습
5. 최신 모델로 재배포

---

## 🧠 주요 기능

### ✔ 시계열 예측 모델
- LSTM 기반 전력 소비량 예측
- 다중 건물 데이터를 통합 학습 (global model)

### ✔ 전처리 자동화
- 시간 기반 feature engineering (sin/cos encoding)
- lag / rolling feature 생성
- building 정보 merge

### ✔ 모델 평가
- RMSE, MAE, MAPE 계산
- threshold 기반 성능 판단

### ✔ 자동 재학습 (MLOps 핵심)
- 성능 저하 시 `retrain.py` 자동 실행
- 모델 및 scaler 업데이트

### ✔ REST API
- FastAPI 기반 `/upload` 엔드포인트
- CSV 업로드 → 예측 → 평가 → 재학습까지 자동 처리

---

## 📂 프로젝트 구조

project/
├── backend/
├── dataset/
├── artifacts/
├── preprocessing.py
├── train.py
├── predict.py
├── evaluate.py
├── retrain.py
└── README.md

---

## ⚙️ 실행 방법

### 1️⃣ 환경 설정
pip install -r requirements.txt

### 2️⃣ 모델 학습
python train.py

### 3️⃣ 서버 실행
uvicorn backend.main:app --reload

---

## 🔄 전체 파이프라인

CSV Upload → predict → evaluate → retrain → model update

---

## 📊 성능 지표
- RMSE
- MAE
- MAPE

---

## 💬 한 줄 요약
시계열 예측 + 자동 재학습까지 포함한 End-to-End MLOps 파이프라인 구현 프로젝트
