## ⚡ 풍력 발전량 예측 ML 모델 구현

### 📝 프로젝트 소개

**목표**: 주어진 시간별 기후 데이터를 기반으로 풍력 발전기의 시간당 발전량(`Generation_Amount`)을 예측하기

---

### 팀 구성

|ML Model|Preprocessing|
|:---:|:---:|
| <img src="https://github.com/Ea3124.png" width="120"> | <img src="https://github.com/mangsgi.png" width="120"> |
|[@이승재](https://github.com/Ea3124)|[@김명석](https://github.com/mangsgi)|
|leesj6717@gmail.com| ms010901@gmail.com |

---

### 📝 프로젝트 개요

* **목표**: 시간별 기후 데이터를 기반으로 풍력 발전기의 시간당 발전량(`Generation_Amount`)을 예측합니다.
* **주어진 데이터**:

  * 발전기 센서 데이터 :
    - 20200201~20200208.csv
    - 20200209~20200215.csv
    - 20200216~20200222.csv
  * ECMWF 기후 데이터 :
    - ECMWF_Climate_Data_Training.csv (2020-02-01 ~ 2020-02-22)
    - ECMWF_Climate_Data_Testing_public.csv (2020-02-01 ~ 2020-02-25)
    - ECMWF_Climate_Data_Testing_private.csv (2020-02-01 ~ 2020-02-29)

---

### 파일 구조

```
.
├── README.md           
├── answer                          # 실제 정답 csv 파일
├── answer.py                       # 정답과 예측 비교
├── dataset                         # 기후, 발전량 dataset
├── img                             # README imgs
├── main.py                         # 전체 코드 flow
├── ml_method                       # ml model관련 사용 함수들
├── preprocessing                   # 전처리 사용 함수들
├── submission_private.csv          # 예측결과(private)
├── submission_public.csv           # 예측결과(public)
└── xgb_final_model.pkl             # optuna로 최적화된 xgb model
```

---

### 🛠️ 데이터 전처리 과정

#### 1. **발전기 데이터 처리**

* 3개 구간의 발전기 데이터를 통합하고, 이상치를 제거
* `Generation_Amount`는 시간별로 **sum**, 나머지 수치는 **mean**으로 resample 
    * 원래는 분당 이었으나, 제출 파일 형식에 맞게 시간당으로 조정
* 최종적으로 528개의 시간 단위 데이터 확보.

#### 2. **기후 데이터 처리**

* ECMWF 기후 데이터(Train)를 시계열 및 위치 기준으로 정렬.
* 강수량은 누적값 → 시간당 변화량으로 변환했으나, 유의미한 판단 기준이 되지 않아 제거
* 지점 보간(interpolation)을 통해, 25개 지점의 기후데이터를, 발전기 위치 한 곳으로 모아 기후 조건 추정
* 특성 공학을 사용하여, 기온과 기압으로 고도 계산, 고도별 풍속 및 풍향 유도 변수 생성.
    - [Log wind profile](https://en.wikipedia.org/wiki/Log_wind_profile)
    - [Barometric formula](https://en.wikipedia.org/wiki/Barometric_formula)
    - [Wind profile power law](https://en.wikipedia.org/wiki/Wind_profile_power_law)


---

### 🔍 상관관계 분석 결과

* 발전량(`Generation_Amount`)과 가장 높은 양의 상관 변수는 `Sensor_Wind_Speed`, `Altitude Wind Speed` 등.
* 하지만 `Sensor_Wind_Speed`는 테스트셋에서는 주어지지 않기 때문에 대체 변수로, 기후데이터에서 특성공학을 통해 생성한 `Altitude Wind Speed`등 을 주요 입력 피처로 사용.

---

### 🤖 모델링 및 성능 비교

| 모델            | MAE (Mean Absolute Error) |
| ------------- | ------------------------- |
| Random Forest | **328.20**                |
| XGBoost       | 364.04                    |
| LightGBM      | 375.30                    |
| ExtraTrees    | 353.67                    |

* 이후 `log1p` 변환으로 타깃을 스케일링한 결과, **MAE 개선**:

  * `RandomForest (log1p)` → **317.91**
  * 단, 아래의 optuna성능으로 인해 XGBOOST로 대체

---

### ⚙️ 하이퍼파라미터 최적화 (Optuna)

* XGBoost에 대해 Optuna로 튜닝한 결과:

```json
"XGBoost": {
  "mae": 295.44,
  "params": {
    "colsample_bytree": 0.76,
    "learning_rate": 0.13,
    "max_depth": 11,
    "n_estimators": 152,
    "subsample": 0.999
  }
}
```

* 튜닝 결과, **기존보다 약 10% 이상 MAE 개선**

---

### 📦 모델 저장 및 추론

* 최적의 XGBoost 모델을 전체 데이터로 재학습 후 `.pkl`로 저장.
* 테스트셋(`public`, `private`)은 날짜 필터링 후 (23–25일, 27–29일) 추론 수행.
* 예측 결과는 `'시간'` 컬럼을 기준으로 `DataFrame` 형태로 저장 가능.

---

### 🔍 실제 정답과 비교 결과

| 구분            | MAE (Mean Absolute Error) |
| ------------- | ------------------------- |
| public        | **435.0549**              |
| private       | **199.9395**              |

- private에서 더 좋은 예측이 되었던 걸 확인 할 수 있었다.
- 다만, 그래프 개형으로는 public이 더 우세한 것을 확인
![예측 결과 그래프(public)](/img/public.png)
![예측 결과 그래프(private)](/img/private.png)
---

### ✅ 결론

* 테스트셋에 발전기 센서 정보가 없는 상황에서도, **기후 변수만으로 조금의 예측은 가능**
* 다만, 기후데이터에서 특성공학으로 만들어 낸 풍속으로는, 정확도가 부족해 dataset자체의 한계가 좀 컸다.