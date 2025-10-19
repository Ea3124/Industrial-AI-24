# -*- coding: utf-8 -*-
"""
메인 코드
"""

import pandas as pd
from preprocessing.generator_preprocessing import generator_preprocessing
from preprocessing.climate_train_preprocessing import process_climate_data
from ml_method.ml_model_select import predict_generation_from_test
from dl_method.lstm import predict_with_saved_lstm
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def load_csv(filename, folder='./dataset', encoding='cp949'):
    path = f'{folder}/{filename}'
    return pd.read_csv(path, encoding=encoding, low_memory=False)

generator_bottom = load_csv('20200201~20200208.csv')
generator_mid = load_csv('20200209~20200215.csv')
generator_top = load_csv('20200216~20200222.csv')
climate_train = load_csv('ECMWF_Climate_Data_Training.csv')
climate_public = load_csv('ECMWF_Climate_Data_Testing_public.csv')
climate_private = load_csv('ECMWF_Climate_Data_Testing_private.csv')

# 발전기 데이터 컬럼 정의
eng_col = [
    'Date',                         # 날짜
    'Generation_Amount',            # 발전량
    'Sensor_Wind_Speed',            # 센서 풍속
    'Generator_Current_A',          # 발전기 전류 A
    'Generator_Current_B',          # 발전기 전류 B
    'Generator_Current_C',          # 발전기 전류 C
    'Gear_Oil_Temperature',         # 기어 오일 온도
    'Generator_Output',             # 발전기 출력
    'Generator_Speed',              # 발전기 속도   
    'Internal_Temperature',         # 내부 온도
    'Coolant_Temperature',          # 냉각수 온도
    'Hydraulic_Oil_Temperature',    # 하이드로릭 오일 온도
    'Pitch_Pressure',               # 피치 압력
    'Rotor_Speed',                  # 로터 속도
    'Winding_Temperature_A',        # 권선 온도 A
    'Winding_Temperature_B',        # 권선 온도 B
    'Winding_Temperature_C',        # 권선 온도 C
    'Transformer_Temperature_BUS',  # 변압기 온도 BUS
    'Transformer_Temperature_A',    # 변압기 온도 A
    'Transformer_Temperature_B',    # 변압기 온도 B
    'Transformer_Temperature_C',    # 변압기 온도 C
    'Generator_Voltage_A',          # 발전기 전압 A
    'Generator_Voltage_B',          # 발전기 전압 B
    'Generator_Voltage_C',          # 발전기 전압 C
    'Generator_Running_State'       # 발전기 실행 상태
]
for df in [generator_bottom, generator_mid, generator_top]:
    df.columns = eng_col

# 시간 순서대로 세 개의 데이터프레임을 합치기 (3상 발전기 데이터)
generator_all = pd.concat([generator_bottom, generator_mid, generator_top], ignore_index=True)

# 발전량 전처리 (1시간 단위, 발전량은 sum, 풍속은 mean)
df_hourly = generator_preprocessing(generator_all, plot=False)

# 풍향 전처리
interp_point = (126.834765, 33.4396799)

climate_processed = process_climate_data(climate_train, interp_point)
# print(climate_processed.info())

# ----------요구사항 해결----------- #

# public 전처리
climate_processed_pub = process_climate_data(climate_public, interp_point)
# print(climate_processed_pub.info())

# private 전처리
climate_processed_pri = process_climate_data(climate_private, interp_point)
# print(climate_processed_pri.info())

# 23일 ~ 25일
pub_filtered = climate_processed_pub[
    (climate_processed_pub['time'] >= datetime(2020, 2, 23)) &
    (climate_processed_pub['time'] <= datetime(2020, 2, 25, 23))
]
# 27일 ~ 29일
pri_filtered = climate_processed_pri[
    (climate_processed_pri['time'] >= datetime(2020, 2, 27)) &
    (climate_processed_pri['time'] <= datetime(2020, 2, 29, 23))
]

# public 예측
pub_result = predict_generation_from_test(pub_filtered)
# pub_result = predict_with_saved_lstm(pub_filtered)
pub_result.to_csv('submission_public.csv', index=False)
print("=== public 예측 저장 완료: submission_public.csv ===")

# private 예측
pri_result = predict_generation_from_test(pri_filtered)
# pri_result = predict_with_saved_lstm(pri_filtered)
pri_result.to_csv('submission_private.csv', index=False)
print("=== private 예측 저장 완료: submission_private.csv ===")
