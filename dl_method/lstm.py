# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ml_method에서 데이터 준비 함수 가져오기
from ml_method.ml_model_select import prepare_merged_data

# 재현성을 위한 랜덤 시드 설정
tf.random.set_seed(42)
np.random.seed(42)

LOOK_BACK = 24 # 시계열 look-back
MODEL_DIR = 'dl_method/models' # 모델 저장 디렉토리

def create_dataset(X_data, y_data, look_back=LOOK_BACK):
    """
    시계열 데이터셋 생성 함수
    """
    Xs, ys = [], []
    for i in range(len(X_data) - look_back):
        v = X_data[i:(i + look_back)]
        Xs.append(v)
        ys.append(y_data[i + look_back])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    """
    LSTM 모델 구성 함수
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def train_and_save_lstm(df_hourly, climate_train):
    """
    LSTM 모델을 학습하고 필요한 파일(모델, 스케일러, 초기값)을 저장합니다.
    """
    print("=== LSTM 모델 학습을 시작합니다... ===")
    
    # 1. 학습 데이터 준비
    X_train_full, y_train_full = prepare_merged_data(df_hourly, climate_train, use_predicted_sensor=True)
    
    # 2. 데이터 스케일링
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_full)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_full.values.reshape(-1, 1))

    # 3. 학습 데이터셋 생성
    X_train, y_train = create_dataset(X_train_scaled, y_train_scaled.flatten(), look_back=LOOK_BACK)

    if X_train.shape[0] == 0:
        raise ValueError(f"{LOOK_BACK} 기간으로 데이터셋을 생성하기에 학습 데이터가 부족합니다.")

    # 4. LSTM 모델 빌드 및 학습
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])

    # 5. 파일 저장 디렉토리 생성
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 6. 파일 저장
    model.save(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, 'lstm_scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'lstm_scaler_y.pkl'))
    
    initial_input = X_train_scaled[-LOOK_BACK:]
    np.save(os.path.join(MODEL_DIR, 'lstm_initial_input.npy'), initial_input)
    
    joblib.dump(X_train_full.columns, os.path.join(MODEL_DIR, 'lstm_feature_columns.pkl'))

    print(f"=== 모델 및 관련 파일이 {MODEL_DIR}에 저장되었습니다. ===")

def predict_with_saved_lstm(climate_test_df):
    """
    저장된 LSTM 모델과 파일들을 불러와 발전량을 예측합니다.
    """
    # 1. 파일 로드
    model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
    scaler_X_path = os.path.join(MODEL_DIR, 'lstm_scaler_X.pkl')
    scaler_y_path = os.path.join(MODEL_DIR, 'lstm_scaler_y.pkl')
    initial_input_path = os.path.join(MODEL_DIR, 'lstm_initial_input.npy')
    feature_columns_path = os.path.join(MODEL_DIR, 'lstm_feature_columns.pkl')

    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    input_sequence = np.load(initial_input_path)
    feature_columns = joblib.load(feature_columns_path)

    # 2. 테스트 데이터 준비
    test_features = climate_test_df[feature_columns]
    test_scaled = scaler_X.transform(test_features)
    
    predictions_scaled = []

    # 3. 예측 루프
    for i in range(len(test_scaled)):
        current_input_reshaped = np.reshape(input_sequence, (1, LOOK_BACK, input_sequence.shape[1]))
        pred_scaled = model.predict(current_input_reshaped, verbose=0)
        predictions_scaled.append(pred_scaled[0, 0])
        
        new_input = test_scaled[i, :].reshape(1, -1)
        input_sequence = np.vstack([input_sequence[1:], new_input])

    # 4. 예측 결과 역스케일링
    y_pred = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

    # 5. 결과 데이터프레임 생성
    result = climate_test_df[['time']].copy()
    result.rename(columns={'time': '시간'}, inplace=True)
    result['발전량'] = y_pred.flatten()

    return result

def load_csv(filename, folder='./dataset', encoding='cp949'):
    path = f'{folder}/{filename}'
    return pd.read_csv(path, encoding=encoding, low_memory=False)

if __name__ == '__main__':
    from preprocessing.generator_preprocessing import generator_preprocessing
    from preprocessing.climate_train_preprocessing import process_climate_data

    print("=== 데이터를 로드하고 전처리를 시작합니다... ===")
    generator_bottom = load_csv('20200201~20200208.csv')
    generator_mid = load_csv('20200209~20200215.csv')
    generator_top = load_csv('20200216~20200222.csv')
    climate_train_raw = load_csv('ECMWF_Climate_Data_Training.csv')

    eng_col = [
        'Date', 
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
        'Generator_Voltage_C', 
        'Generator_Running_State'
    ]
    for df in [generator_bottom, generator_mid, generator_top]:
        df.columns = eng_col

    generator_all = pd.concat([generator_bottom, generator_mid, generator_top], ignore_index=True)
    df_hourly = generator_preprocessing(generator_all, plot=False)
    
    interp_point = (126.834765, 33.4396799)
    climate_processed = process_climate_data(climate_train_raw, interp_point)
    print("=== 데이터 준비 완료 ===")

    # 학습 및 모델 저장 실행
    train_and_save_lstm(df_hourly, climate_processed)