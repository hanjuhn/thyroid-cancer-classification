# Thyroid-Cancer-Classification | 갑상선 암 여부 예측

## 🌟 Project Overview | 프로젝트 개요

Thyroid-Cancer-Classification은 갑상선 암 여부를 예측하기 위한 머신러닝 기반의 이진 분류 모델입니다. LGBM, XGBoost, CatBoost 모델을 스태킹 앙상블 방식으로 결합하였으며 각 모델의 하이퍼파라미터는 Optuna를 활용해 자동으로 최적화하였습니다.

---

## 📁 Key Directories and Files | 주요 디렉토리 및 파일

- `src/utils.py`: 데이터 로딩, 전처리 및 시드 고정 함수 정의
- `src/objectives.py`: 모델별 Optuna 목적 함수 정의 및 하이퍼파라미터 최적화 실행
- `src/model.py`: Stacking 모델 학습, 예측 수행 및 결과 저장
- `main.py`: 전체 파이프라인 실행을 위한 메인 스크립트
