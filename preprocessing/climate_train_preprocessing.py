# -*- coding: utf-8 -*-
"""
climate_train_preprocessing
"""

import pandas as pd
import numpy as np
import xarray as xr

# 상수 정의
R = 287.05  # 건조 공기의 기체 상수 (J/(kg·K))
g = 9.80665  # 중력 가속도 (m/s²)

def calculate_altitude(temperature, sea_level_pressure, surface_pressure):
    return ((R * temperature) / g) * np.log(sea_level_pressure / surface_pressure)

def calculate_z0(V_10, V_100, eps=1e-5):
    ratio = V_100 / (V_10 + eps)  # divide-by-zero 보호
    denominator = 1 - ratio
    denominator = denominator if abs(denominator) > eps else eps  # 또 한 번 보호

    ln_z0 = (np.log(100) - ratio * np.log(10)) / denominator
    z0 = np.exp(ln_z0)
    return z0

def calculate_wind_speed(V_10, z0, target_height):
    return V_10 * (np.log(target_height / z0) / np.log(10 / z0))

def calculate_wind_components(U_10, V_10, z0, target_height):
    factor = np.log(target_height / z0) / np.log(10 / z0)
    return U_10 * factor, V_10 * factor

def calculate_wind_direction(U, V):
    rad = np.arctan2(U, V)
    deg = np.degrees(rad)
    return (deg + 360) % 360

def convert_precip_to_hourly(df, cum_col='Total precipitation'):
    df = df.sort_values(['lon', 'lat', 'time']).copy()
    df['prev_precip'] = df.groupby(['lon', 'lat'])[cum_col].shift(1).fillna(0)
    df['diff'] = df[cum_col] - df['prev_precip']
    df['Total precipitation'] = df['diff'].where(df['diff'] >= 0, df[cum_col])
    return df.drop(columns=['prev_precip', 'diff'])

def align_last_25_times(df, block_size: int = 200):
    df = df.copy()
    n_rows = len(df)
    tail_len = 25
    for start in range(0, n_rows, block_size):
        pivot_idx = start - 1 + (block_size - tail_len)
        if pivot_idx >= n_rows:
            break
        pivot_time = df.at[pivot_idx, 'time']
        end_idx = min(start + block_size, n_rows) - 1
        df.loc[pivot_idx + 1:end_idx, 'time'] = pivot_time
    return df

def interpolate_point(ds: xr.Dataset, lon_pt: float, lat_pt: float, method: str = 'linear') -> pd.DataFrame:
    ds_i = ds.interp(lon=lon_pt, lat=lat_pt, method=method)
    df_i = ds_i.to_dataframe().reset_index()
    df_i['lon_pt'] = lon_pt
    df_i['lat_pt'] = lat_pt
    return df_i

def process_climate_data(climate_train: pd.DataFrame,
                         interp_point: tuple[float, float],
                         drop_precip=True) -> pd.DataFrame:
    """
    기후 데이터를 전처리하고 특정 지점에서 보간 및 고도/풍속/풍향 계산까지 수행

    Parameters
    ----------
    climate_train : pd.DataFrame
        원시 기후 데이터 ('time', 'lon', 'lat', 'parameterName', 'value' 포함)
    interp_point : tuple
        (lon, lat) 위치에서 보간 수행할 지점
    drop_precip : bool
        누적 강수량 컬럼 삭제 여부

    Returns
    -------
    pd.DataFrame
        보간 및 계산이 완료된 DataFrame
    """
    df = climate_train.dropna(how='all').copy()
    df['time'] = pd.to_datetime(df['time'])
    df = align_last_25_times(df)
    
    df_agg = (
        df.groupby(["time", "lon", "lat", "parameterName"])
          .mean(numeric_only=True)
          .reset_index()
    )

    df_resampled = (
        df_agg.set_index('time')
              .groupby(["lon", "lat", "parameterName"])
              .resample('h')['value']
              .mean(numeric_only=True)
              .reset_index()
    )

    df_pivot = df_resampled.pivot(index=["time", "lon", "lat"], columns="parameterName", values="value").reset_index()

    if 'Total precipitation' in df_pivot.columns:
        df_pivot = convert_precip_to_hourly(df_pivot, 'Total precipitation')

    if drop_precip and 'Total precipitation' in df_pivot.columns:
        df_pivot = df_pivot.drop(columns=['Total precipitation'])

    # Xarray 변환 & 보간
    ds = df_pivot.set_index(['time', 'lon', 'lat']).to_xarray()
    interp_df = interpolate_point(ds, lon_pt=interp_point[0], lat_pt=interp_point[1], method='linear')
    interp_df = interp_df.drop(columns=['lon_pt', 'lat_pt'])

    # 계산 파트
    df = interp_df.copy()

    df['Air Density'] = df['Surface pressure'] / (R * df['2 metre temperature'])
    df['Altitude'] = calculate_altitude(df['2 metre temperature'], df['Mean sea level pressure'], df['Surface pressure'])
    df['10m wind speed'] = np.sqrt(df['10 metre U wind component']**2 + df['10 metre V wind component']**2)
    df['100m wind speed'] = np.sqrt(df['100 metre U wind component']**2 + df['100 metre V wind component']**2)

    df['surface zodo len'] = df.apply(lambda row: calculate_z0(row['10m wind speed'], row['100m wind speed']), axis=1)
    df['Altitude Wind Speed'] = df.apply(lambda row: calculate_wind_speed(row['10m wind speed'], row['surface zodo len'], row['Altitude']), axis=1)

    df['U component at Altitude'], df['V component at Altitude'] = zip(*df.apply(
        lambda row: calculate_wind_components(row['10 metre U wind component'], row['10 metre V wind component'],
                                              row['surface zodo len'], row['Altitude']), axis=1))

    df['Altitude Wind Direction'] = df.apply(lambda row: calculate_wind_direction(row['U component at Altitude'], row['V component at Altitude']), axis=1)
    df['Altitude Wind Direction_sin'] = np.sin(np.radians(df['Altitude Wind Direction']))
    df['Altitude Wind Direction_cos'] = np.cos(np.radians(df['Altitude Wind Direction']))

    df['10m wind direction'] = df.apply(lambda row: calculate_wind_direction(row['10 metre U wind component'], row['10 metre V wind component']), axis=1)
    df['100m wind direction'] = df.apply(lambda row: calculate_wind_direction(row['100 metre U wind component'], row['100 metre V wind component']), axis=1)

    return df