# -*- coding: utf-8 -*-
"""
메인코드
"""

import pandas as pd
from preprocessing.generator_preprocessing import generator_preprocessing
from preprocessing.climate_train_preprocessing import process_climate_data
from ml_method.ml_model_select import predict_generation_from_test
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

# 시간순으로 세 개의 데이터프레임을 합치기
generator_all = pd.concat([generator_bottom, generator_mid, generator_top], ignore_index=True)

# 발전량 전처리(발전량은 sum, 풍속은 mean)
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
pub_result.to_csv('submission_public.csv', index=False)
print("✅ public 예측 저장 완료: submission_public.csv")

# private 예측
pri_result = predict_generation_from_test(pri_filtered)
pri_result.to_csv('submission_private.csv', index=False)
print("✅ private 예측 저장 완료: submission_private.csv")
