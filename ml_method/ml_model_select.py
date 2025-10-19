# -*- coding: utf-8 -*-
"""
데이터 삭제하는 방식의 전처리 진행
"""

import joblib
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import numpy as np
from datetime import datetime

def prepare_merged_data(df_hourly, climate_processed, use_predicted_sensor=False):
    """
    발전기 데이터와 기후 데이터를 병합하는 함수
    """
    df = df_hourly.reset_index().merge(
        climate_processed,
        how='inner',
        left_on='Date',
        right_on='time'
    )

    # 센서 관련 열 제거
    eng_col_remove = [
        'Date', 
        'time', 
        'lon', 
        'lat', 
        'Generator_Running_State',
        'Generation_Amount',
        'Sensor_Wind_Speed',
        'Generator_Current_A', 
        'Generator_Current_B', 
        'Generator_Current_C',
        'Gear_Oil_Temperature', 
        'Generator_Output', 
        'Generator_Speed',
        'Internal_Temperature', 
        'Coolant_Temperature', 
        'Hydraulic_Oil_Temperature',
        'Pitch_Pressure', 
        'Rotor_Speed',
        'Winding_Temperature_A', 
        'Winding_Temperature_B', 
        'Winding_Temperature_C',
        'Transformer_Temperature_BUS', 
        'Transformer_Temperature_A', 
        'Transformer_Temperature_B', 
        'Transformer_Temperature_C',
        'Generator_Voltage_A', 
        'Generator_Voltage_B', 
        'Generator_Voltage_C'
    ]

    y = df['Generation_Amount']

    if use_predicted_sensor:
        cols_to_drop = eng_col_remove
    else:
        cols_to_drop = eng_col_remove + ['Predicted_Sensor_Wind_Speed']

    X = df.drop(columns=cols_to_drop, errors='ignore')  # errors='ignore'로 안전하게
    return X, y

def train_and_evaluate_model(model, X, y, name='Model'):
    """
    모델 학습 및 성능 평가 함수
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False  # 시계열 유지
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"{name} MAE: {mae:.4f}")
    return model, mae

def train_and_evaluate(X, y, name='Model'):
    """
    모델 학습 및 성능 평가 함수 (모델 고정)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False  # 시계열은 시간 순서 유지
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    print(f"{name} MAE: {mae:.4f}")
    return model, mae

def plot_feature_importance(model, feature_names, top_n=19):
    """
    특성 중요도 그래프 함수
    """
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    top_features = sorted_idx[:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[top_features][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in top_features][::-1])
    plt.xlabel('Importance')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.show()

def get_top_features(model, feature_names, top_n=10):
    """
    특성 중요도 상위 top_n개 컬럼 반환 함수
    """
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    return [feature_names[i] for i in sorted_idx[:top_n]]


# # ---------- ① Model A: RandomForest ----------
# X_a, y_a = prepare_merged_data(df_hourly, climate_processed, use_predicted_sensor=True)
# model_rt, mae_rt = train_and_evaluate(X_a, y_a, name='Model A (No Sensor_Wind_Speed)')

# # ---------- ② Model B: XGBoost ----------
# xgb_model = XGBRegressor(n_estimators=100, random_state=42)
# model_xgb, mae_xgb = train_and_evaluate_model(xgb_model, X_a, y_a, name='Model B (XGBoost)')

# # ---------- ③ Model C: LightGBM ----------
# lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
# model_lgbm, mae_lgbm = train_and_evaluate_model(lgbm_model, X_a, y_a, name='Model C (LightGBM)')

# # ---------- ③ Model D: ExtraTree ----------
# et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
# model_et, mae_et = train_and_evaluate_model(et_model, X_a, y_a, name='Model D (ExtraTrees)')

# # plot_feature_importance(model_a, X_a.columns)
# print("\n📊 MAE 비교")
# print(f"RandomForest : {mae_rt:.4f}")
# print(f"XGBoost      : {mae_xgb:.4f}")
# print(f"LightGBM     : {mae_lgbm:.4f}")
# print(f"ExtraTree    : {mae_et:.4f}")


# Target log 변환
# y_log = np.log1p(y_a)

# # 훈련/검증 나누기 (시계열 순서 유지)
# X_train, X_val, y_train_log, y_val_log = train_test_split(X_a, y_log, test_size=0.2, shuffle=False)

def objective_rf(trial):
    """
    Random Forest 모델 학습 함수
    """
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        max_depth=trial.suggest_int('max_depth', 3, 20),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        random_state=42
    )
    model.fit(X_train, y_train_log)
    y_pred = np.expm1(model.predict(X_val))
    return mean_absolute_error(np.expm1(y_val_log), y_pred)

def objective_xgb(trial):
    """
    XGBoost 모델 학습 함수
    """
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        max_depth=trial.suggest_int('max_depth', 3, 15),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train_log)
    y_pred = np.expm1(model.predict(X_val))
    return mean_absolute_error(np.expm1(y_val_log), y_pred)

def objective_lgbm(trial):
    """
    LightGBM 모델 학습 함수
    """
    model = LGBMRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        max_depth=trial.suggest_int('max_depth', 3, 15),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
        num_leaves=trial.suggest_int('num_leaves', 20, 150),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        random_state=42
    )
    model.fit(X_train, y_train_log)
    y_pred = np.expm1(model.predict(X_val))
    return mean_absolute_error(np.expm1(y_val_log), y_pred)

def objective_et(trial):
    """
    ExtraTrees 모델 학습 함수
    """
    model = ExtraTreesRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        max_depth=trial.suggest_int('max_depth', 3, 20),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        random_state=42
    )
    model.fit(X_train, y_train_log)
    y_pred = np.expm1(model.predict(X_val))
    return mean_absolute_error(np.expm1(y_val_log), y_pred)


# Random Forest
# study_rf = optuna.create_study(direction='minimize')
# study_rf.optimize(objective_rf, n_trials=300)
# print("✅ RF Best MAE:", study_rf.best_value)

# # XGBoost
# study_xgb = optuna.create_study(direction='minimize')
# study_xgb.optimize(objective_xgb, n_trials=300)
# print("✅ XGB Best MAE:", study_xgb.best_value)

# # LightGBM
# study_lgbm = optuna.create_study(direction='minimize')
# study_lgbm.optimize(objective_lgbm, n_trials=300)
# print("✅ LGBM Best MAE:", study_lgbm.best_value)

# # ExtraTrees
# study_et = optuna.create_study(direction='minimize')
# study_et.optimize(objective_et, n_trials=300)
# print("✅ ET Best MAE:", study_et.best_value)


# results = {
#     "RandomForest": {
#         "mae": study_rf.best_value,
#         "params": study_rf.best_params
#     },
#     "XGBoost": {
#         "mae": study_xgb.best_value,
#         "params": study_xgb.best_params
#     },
#     "LightGBM": {
#         "mae": study_lgbm.best_value,
#         "params": study_lgbm.best_params
#     },
#     "ExtraTrees": {
#         "mae": study_et.best_value,
#         "params": study_et.best_params
#     }
# }

# print("📊 Best MAEs:", results)

# Best MAEs: {
# 'RandomForest': {'mae': 299.6347434097198, 'params': {'n_estimators': 287, 'max_depth': 3, 'min_samples_split': 6, 'min_samples_leaf': 6, 'max_features': None}}, 
# 'XGBoost': {'mae': 295.4434350211665, 'params': {'n_estimators': 152, 'max_depth': 11, 'learning_rate': 0.13200740123317442, 'subsample': 0.9990413522421722, 'colsample_bytree': 0.7618993936494004}}, 
# 'LightGBM': {'mae': 300.0865091744985, 'params': {'n_estimators': 72, 'max_depth': 13, 'learning_rate': 0.014357764389799507, 'num_leaves': 26, 'subsample': 0.9722508599365528, 'colsample_bytree': 0.871686254214325}}, 
# 'ExtraTrees': {'mae': 300.94075733053677, 'params': {'n_estimators': 244, 'max_depth': 3, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': None}}}

# import joblib
# from xgboost import XGBRegressor

# # ① 최적 params
# best_params_xgb = {
#     'colsample_bytree': 0.7618993936494004,
#     'learning_rate': 0.13200740123317442,
#     'max_depth': 11,
#     'n_estimators': 152,
#     'subsample': 0.9990413522421722
# }

# # ② 전체 학습용 데이터
# X_all, y_all = prepare_merged_data(df_hourly, climate_processed, use_predicted_sensor=True)

# # ③ 재학습
# final_model_xgb = XGBRegressor(**best_params_xgb, random_state=42)
# final_model_xgb.fit(X_all, y_all)

# # ④ 모델 저장
# joblib.dump(final_model_xgb, 'xgb_final_model.pkl')
# print("✅ 모델 저장 완료: xgb_final_model.pkl")


def predict_generation_from_test(climate_test_df, model_path='xgb_final_model.pkl'):
    """
    저장된 모델을 불러와, test 기후 데이터에 대해 발전량을 예측합니다.
    
    Parameters
    ----------
    climate_test_df : pd.DataFrame
        23~25일 기후 전처리 데이터
    model_path : str
        저장된 모델 경로
    
    Returns
    -------
    pd.DataFrame
        시간 & 예측 발전량 포함된 DataFrame
    """
    # 1. 모델 로드
    model = joblib.load(model_path)

    # 2. 발전량 학습용 컬럼 제외
    df_input = climate_test_df.drop(columns=['lon', 'lat', 'time'], errors='ignore')

    # 3. 예측
    df_input = df_input.drop(columns=['surface zodo len'], errors='ignore')
    y_pred = model.predict(df_input)

    # 4. 결과 df
    result = climate_test_df[['time']].copy()
    result.rename(columns={'time': '시간'}, inplace=True)
    result['발전량'] = y_pred

    return result



