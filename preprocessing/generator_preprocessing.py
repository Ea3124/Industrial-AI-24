# -*- coding: utf-8 -*-
""" data_preprocessing.ipynb
데이터 삭제하는 방식의 전처리 진행
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.generater_preprocesing_func.remove_outlier import plot_generator_features, boxplot_generator_features, plot_generator_features_exclude_outliers, filter_generator_outliers
from preprocessing.generater_preprocesing_func.interpolation import plot_correlation_heatmap, interpolate_missing_values_time

def generator_preprocessing(generator_all, plot=False):
    """
    풍력 발전기 데이터 전처리 파이프라인 함수

    Parameters:
    - generator_all (pd.DataFrame): 원본 데이터 (Date 포함)
    - plot (bool): 중간 시각화를 출력할지 여부

    Returns:
    - df_hourly (pd.DataFrame): 전처리 및 리샘플링된 데이터 (1시간 단위)
    """
    # 1. datetime 처리
    df = generator_all.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').set_index('Date')
    
    if plot:
            print("📈 필터링 전 시각화:")
            plot_generator_features_exclude_outliers(df)

    # 2. 이상치 제거
    filtered = filter_generator_outliers(df)

    if plot:
        print("📈 필터링 후 시각화:")
        plot_generator_features(filtered)
        boxplot_generator_features(filtered)
        plot_correlation_heatmap(filtered)

    # 3. 결측치 보간 (Sensor_Wind_Speed 기준, 상관계수 0.7 이상만)
    interpolated_df = interpolate_missing_values_time(
        filtered,
        reference='Sensor_Wind_Speed',
        corr_threshold=0.7
    )

    if plot:
        print("📊 보간 후 시각화:")
        plot_generator_features(interpolated_df)

    # 4. 수치형 컬럼: 'Generation_Amount'는 sum, 나머지는 mean
    df_numeric = interpolated_df.select_dtypes(include='number')
    agg_dict = {col: 'mean' for col in df_numeric.columns}
    agg_dict['Generation_Amount'] = 'sum'

    df_numeric_resampled = df_numeric.resample('1h').agg(agg_dict)

    # 5. 문자형 컬럼: 상태값 등은 첫 값 유지
    df_strings = interpolated_df.select_dtypes(exclude='number').resample('1h').first()

    # 6. 결합
    df_hourly = pd.concat([df_numeric_resampled, df_strings], axis=1)

    if plot:
        print("1시간 단위 풍속 리샘플링")
        col = 'Sensor_Wind_Speed'

        plt.figure(figsize=(12, 4))
        plt.plot(df_hourly.index, df_hourly[col], marker='o', markersize=3, linestyle='-', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.title(f'Time Series of {col} (Using DatetimeIndex)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_hourly
