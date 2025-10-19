### 딥러닝 방법론 적용

> 기존 전처리 방식에 LSTM 모델을 사용하여 학습한 모델을 통해 추론

<br>

#### 적용 결과

| 구분    | MAE (Mean Absolute Error) |
| ------- | ------------------------- |
| public  | **328.2009**              |
| private | **142.0177**              |

<br>

#### Models 디렉토리 내 파일 설명

1. `lstm_final_model.h5` (모델 파일)

   - 학습한 모델

2. `lstm_scaler_X.pkl` / `lstm_scaler_y.pkl` (스케일러 파일)

   - 학습 때 사용했던 정규화 기준을 저장

3. `lstm_initial_input.npy` (초기 입력 시퀀스)

   - 테스트 기간의 첫 번째 시간을 예측하기 위해, 학습 기간의 마지막 24시간 데이터 저장

4. `lstm_feature_columns.pkl` (특성 이름 목록)

   - 학습 때 사용한 특성(컬럼)들의 순서와 이름을 정확히 기억하기 위해 저장
