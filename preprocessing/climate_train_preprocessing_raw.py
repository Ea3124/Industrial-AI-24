# -*- coding: utf-8 -*-
"""
climate_train_preprocessing, but raw.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import xarray as xr

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def load_csv(filename, folder='./dataset', encoding='cp949'):
    path = f'{folder}/{filename}'
    return pd.read_csv(path, encoding=encoding, low_memory=False)

climate_train = load_csv('ECMWF_Climate_Data_Training.csv')

climate_train = climate_train.dropna(how='all')

# print(climate_train.tail())

# time 컬럼을 datetime 형식으로 변환
climate_train['time'] = pd.to_datetime(climate_train['time'], format='%Y-%m-%d %H:%M')

# 현재는 강수량이 누적시작시간으로 되어있다 -> 매 시간마다로 바꾸기
def align_last_25_times(df, block_size: int = 200, inplace: bool = False):
    """
    각 block_size(기본 200) 행 단위로 잘라,
    블록 내 마지막 25행(176~200번째)의 'time' 값을
    175번째 행의 'time' 값으로 맞춘다.

    Parameters
    ----------
    df : pd.DataFrame
        'time' 컬럼을 포함한 원본 데이터.
    block_size : int, optional
        블록 크기. 기본 200.
    inplace : bool, optional
        True 이면 원본 df를 직접 수정, False 이면 사본 반환. 기본 False.

    Returns
    -------
    pd.DataFrame
        수정된 DataFrame (inplace=False 일 때).
    """
    if not inplace:
        df = df.copy()

    n_rows = len(df)
    # 25 = block_size - 175 (마지막 25행)
    tail_len = 25              

    for start in range(0, n_rows, block_size):
        pivot_idx = start -1 + (block_size - tail_len)  # 175번째(0-base)
        if pivot_idx >= n_rows:
            break  # 남은 행이 175번째보다 짧으면 중단

        # pivot 행의 time
        pivot_time = df.at[pivot_idx, 'time']
        # aa = df.at[175, 'time']

        # print(pivot_time)

        # pivot 다음 행부터 블록 끝까지 time 동일하게 세팅
        end_idx = min(start + block_size, n_rows) - 1
        df.loc[pivot_idx + 1 : end_idx, 'time'] = pivot_time

    return df

# 강수량 time 수정
climate_train_rain_time_ch = align_last_25_times(climate_train)


# 중복된 행 체크
# duplicate_counts = (
#     climate_train_rain_time_ch
#     .groupby(["time", "lon", "lat", "parameterName"])
#     .size()
#     .reset_index(name='count')
#     .query('count > 1')
# )

# print(f"중복된 그룹 수: {len(duplicate_counts)}")
# print(duplicate_counts.head())

# # 중복 항목을 평균값으로 집계
climate_train_agg = climate_train_rain_time_ch.groupby(["time", "lon", "lat", "parameterName"]).mean().reset_index()

# # 데이터를 1시간 단위로 리샘플링
climate_train_agg = (
    climate_train_agg
    .set_index('time')
    .groupby(["lon", "lat", "parameterName"])
    .resample('h')['value']
    .mean(numeric_only=True)
    .reset_index()
)

# 1) 리샘플링된 데이터프레임에서 'value' 열의 결측 개수 확인 - 없음
# print("리샘플링 후 결측 개수 (value):", climate_train_agg['value'].isna().sum())

# # 데이터를 피벗 테이블로 변환
pivot_climate_train = climate_train_agg.pivot(index=["time", "lon", "lat"], columns="parameterName", values="value").reset_index()

# 강수량이 이제 누적합이 아닌, 매시간당 강수량으로 변경
def convert_precip_to_hourly(df, cum_col='Total precipitation'):
    """
    12시간 누적 강수량(cum_col)을 시간당 강수량으로 변환하여
    'precip_hourly' 컬럼에 저장합니다.

    - 같은 (lon, lat) 그룹 내에서 이전 누적값(prev_precip)과 차를 구하고,
    - diff < 0 (리셋 감지) 시에는 diff 대신 현재 cum_col 값을 사용합니다.

    Parameters
    ----------
    df : pd.DataFrame
        'time', 'lon', 'lat', cum_col 컬럼을 가진 데이터프레임
    cum_col : str
        누적 강수량 컬럼명 (기본 'Total precipitation')

    Returns
    -------
    pd.DataFrame
        원본에 'precip_hourly' 컬럼이 추가된 복사본
    """
    df = df.sort_values(['lon', 'lat', 'time']).copy()

    # 이전 누적값
    df['prev_precip'] = (
        df
        .groupby(['lon', 'lat'])[cum_col]
        .shift(1)
        .fillna(0)
    )

    # raw 차분
    df['diff'] = df[cum_col] - df['prev_precip']

    # 음수(diff<0)면 현재 cum_col 값을, 그렇지 않으면 diff를 취함
    df['Total precipitation'] = df['diff'].where(df['diff'] >= 0, df[cum_col])

    # 정리
    df.drop(columns=['prev_precip', 'diff'], inplace=True)
    return df

# kg/m^2 -> mm 로 변경시, 미비함

# 8. 시계열 시각화
# parameter_columns = pivot_climate_train.columns.difference(['time', 'lon', 'lat'])

# col = 'Total precipitation'

# plt.figure(figsize=(12, 4))
# plt.scatter(pivot_climate_train['time'], pivot_climate_train[col], s=10)  # s는 점 크기
# plt.xlabel('Time')
# plt.ylabel(col)
# plt.title(f'Time Series of {col} (Scatter Plot)')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Total precipitation 열을 제거
pivot_climate_train_no_rain = pivot_climate_train.drop(columns=['Total precipitation'])

# 결과 데이터프레임 확인
# print(pivot_climate_train_no_rain.info())

# # # 부분 데이터프레임 확인
# print(pivot_climate_train.iloc[-40:-1])

# -- 여기까지 처음 처리

def plot_climate_features(df,
                          time_col='time',
                          ignore_cols=('lon','lat'),
                          scatter_area=10,
                          columns_per_row=3):
    """
    pivot_climate_train 같은 wide-format DataFrame에서
    time 기반으로 각 변수들의 시계열을 scatter 형태로 한눈에 보여줍니다.

    Parameters
    ----------
    df : pd.DataFrame
        반드시 time_col을 포함하고, 나머지 열이 모두 시계열 변수여야 합니다.
    time_col : str
        시간 축으로 사용할 컬럼명 (기본 'time').
    ignore_cols : tuple of str
        그리지 않을 열들 (예: lon, lat).
    scatter_area : int
        각 점의 크기.
    columns_per_row : int
        한 행에 그릴 subplot 개수.
    """
    # 1) plot 대상 컬럼 목록
    vars_to_plot = [c for c in df.columns
                    if c != time_col and c not in ignore_cols]

    total = len(vars_to_plot)
    rows = math.ceil(total / columns_per_row)

    # 2) figure & axes 생성
    fig, axes = plt.subplots(rows, columns_per_row,
                             figsize=(columns_per_row*4, rows*3),
                             sharex=True)
    # 1차원 array로
    axes = axes.flatten() if total > 1 else [axes]

    # 3) 각 변수마다 scatter 그리기
    for i, var in enumerate(vars_to_plot):
        ax = axes[i]
        ax.scatter(df[time_col], df[var], s=scatter_area, alpha=0.6)
        ax.set_title(var.replace('_',' ').title())
        ax.grid(True)

    # 4) 남는 subplot 숨기기
    for j in range(total, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# pivot_climate_train: 이미 전처리·피벗된 DataFrame
# plot_climate_features(pivot_climate_train,
#                       time_col='time',
#                       ignore_cols=('lon','lat'),
#                       scatter_area=8,
#                       columns_per_row=4)

# -- 내가

# 2) Pandas → Xarray Dataset 변환
ds = (
    pivot_climate_train_no_rain
    .set_index(['time', 'lon', 'lat'])   # ← 이렇게
    .to_xarray()
)

# 3) 보간 함수: 한 점(lon_pt, lat_pt)에 대해 모든 변수 · 모든 시간에 보간 수행
def interpolate_point(ds: xr.Dataset, lon_pt: float, lat_pt: float, method: str = 'linear') -> pd.DataFrame:
    """
    xr.Dataset.interp 으로 lon/lat 보간을 수행하면
    결과는 time 차원만 남은 Dataset이 됩니다.
    이를 DataFrame으로 바꿔 반환합니다.
    """
    # 보간 (time 차원은 자동 유지)
    ds_i = ds.interp(lon=lon_pt, lat=lat_pt, method=method)
    
    # DataFrame으로 변환
    df_i = ds_i.to_dataframe().reset_index()
    # 보간 대상 좌표 정보 컬럼으로 추가하면 구분하기 편합니다
    df_i['lon_pt'] = lon_pt
    df_i['lat_pt'] = lat_pt
    return df_i

# 4) 발전량 지점에 대해 보간
point = [
    (126.834765, 33.4396799),
]

interp_dfs = []
for lon_pt, lat_pt in point:
    df_pt = interpolate_point(ds, lon_pt, lat_pt, method='linear')
    interp_dfs.append(df_pt)

# 5) 결과 합치기
interpolated_all = pd.concat(interp_dfs, ignore_index=True)


# 두 열을 제거
interpolated_df = interpolated_all.drop(columns=['lon_pt', 'lat_pt'])

# 6) 결과 확인
# print(interpolated_df.iloc[0:100])
# print(interpolated_df.info())

# ----여기 두번째

# import math
# import pandas as pd
# import numpy as np

# 상수 정의
R = 287.05  # 건조 공기의 기체 상수 (J/(kg·K))
g = 9.80665  # 중력 가속도 (m/s²)

# 고도 계산 함수
def calculate_altitude(temperature, sea_level_pressure, surface_pressure):
    return ((R * temperature) / g) * np.log(sea_level_pressure / surface_pressure)

# 로그 법칙을 사용하여 표면 조도 길이 z0 계산 함수
def calculate_z0(V_10, V_100):
    ln_z0 = (np.log(100) - (V_100 / V_10) * np.log(10)) / (1 - (V_100 / V_10))
    z0 = np.exp(ln_z0)
    return z0

# 특정 고도에서의 풍속 계산 함수
def calculate_wind_speed(V_10, z0, target_height):
    return V_10 * (np.log(target_height / z0) / np.log(10 / z0))

# 특정 고도에서의 U와 V 성분 계산 함수
def calculate_wind_components(U_10, V_10, z0, target_height):
    factor = np.log(target_height / z0) / np.log(10 / z0)
    U_altitude = U_10 * factor
    V_altitude = V_10 * factor
    return U_altitude, V_altitude

# 풍향 계산 함수
def calculate_wind_direction(U, V):
    rad = np.arctan2(U, V)
    degree = np.degrees(rad)
    degree = (degree + 360) % 360  # 북쪽 기준으로 변환
    return degree

df  = interpolated_df
# df = df.astype(float)

# 밀도 계산
df['Air Density'] = df['Surface pressure'] / (R * df['2 metre temperature'])

# 고도 계산
df['Altitude'] = calculate_altitude(df['2 metre temperature'], df['Mean sea level pressure'], df['Surface pressure'])

# 10m 및 100m에서의 풍속 계산
df['10m wind speed'] = np.sqrt(df['10 metre U wind component']**2 + df['10 metre V wind component']**2)
df['100m wind speed'] = np.sqrt(df['100 metre U wind component']**2 + df['100 metre V wind component']**2)

# 데이터프레임에 새 열 추가 (표면 조도 길이)
df['surface zodo len'] = df.apply(lambda row: calculate_z0(row['10m wind speed'], row['100m wind speed']), axis=1)

# 목표 고도 풍속 구하기
df['Altitude Wind Speed'] = df.apply(lambda row: calculate_wind_speed(row['10m wind speed'], row['surface zodo len'], row['Altitude']), axis=1)


# 고도에서의 U와 V 성분 계산
df['U component at Altitude'], df['V component at Altitude'] = zip(*df.apply(
    lambda row: calculate_wind_components(row['10 metre U wind component'], row['10 metre V wind component'], row['surface zodo len'], row['Altitude']), axis=1))

# 고도에서의 풍향 계산
df['Altitude Wind Direction'] = df.apply(lambda row: calculate_wind_direction(row['U component at Altitude'], row['V component at Altitude']), axis=1)

df['Altitude Wind Direction_sin'] = np.sin(np.radians(df['Altitude Wind Direction']))
df['Altitude Wind Direction_cos'] = np.cos(np.radians(df['Altitude Wind Direction']))

# 풍향 계산 (10m)
df['10m wind direction'] = df.apply(lambda row: calculate_wind_direction(row['10 metre U wind component'], row['10 metre V wind component']), axis=1)

# 풍향 계산 (100m)
df['100m wind direction'] = df.apply(lambda row: calculate_wind_direction(row['100 metre U wind component'], row['100 metre V wind component']), axis=1)

# print(df.info())


# 그래프로 확인
parameter_columns = df.columns.difference(['time'])

col = 'Altitude Wind Speed'

plt.figure(figsize=(12, 4))
plt.scatter(df['time'], df[col], s=10)  # s는 점 크기
plt.scatter(df['time'], df[col], s=10)  # s는 점 크기
plt.xlabel('Time')
plt.ylabel(col)
plt.title(f'Time Series of {col} (Scatter Plot)')
plt.grid(True)
plt.tight_layout()
plt.show()

# # 결과 출력
# wind_df = df.drop(columns=[
#     'surface zodo len',
#     '10m wind speed',
#     '100m wind speed',
#     'U component at Altitude',
#     'V component at Altitude',
#     '10m wind direction',
#     '100m wind direction',
#     ])

# final_columns = ['2 metre temperature', 'Mean sea level pressure', 'Surface pressure', 'Air Density',
#                  'Altitude Wind Speed', 'Altitude Wind Direction']
# wind_df_no_uv = df[final_columns]


# wind_df

# start_time = pd.Timestamp('2020-02-01 00:00:00')

# wind_df['datetime'] = pd.date_range(start=start_time, periods=len(wind_df), freq='H')
# wind_df_no_uv['datetime'] = pd.date_range(start=start_time, periods=len(wind_df_no_uv), freq='H')

# wind_df_no_uv

# plt.figure(figsize=(14, 6))
# sns.scatterplot(x='datetime', y='Altitude Wind Direction', data=wind_df, color='g', alpha=1, s=5)
# mean_values = wind_df.mean()
# mean_values

# # wind_df.to_csv('./dataset/wind_df0813_yes_sc.csv', index=False)
# # wind_df_no_uv.to_csv('./dataset/wind_df0807_no.csv', index=False)

# plt.figure(figsize=(14, 6))
# sns.lineplot(x='datetime', y='Altitude Wind Speed', data=wind_df, color='g', alpha=1)







