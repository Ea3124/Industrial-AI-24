# -*- coding: utf-8 -*-
""" data_preprocessing.ipynb
데이터 삭제하는 방식의 전처리 진행
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error


# 극단 제외 후, 서로 연관성 비교
def plot_correlation_heatmap(
    df,
    cols=None,
    figsize=(14, 10),
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    annot=False,
    square=True,
    title='Correlation Heatmap'
):
    """
    숫자형 컬럼들 간의 상관계수 히트맵을 그려줍니다.

    Parameters
    ----------
    df : pd.DataFrame
        분석할 데이터프레임.
    cols : list of str, optional
        상관계수를 계산할 컬럼 리스트. None이면 df의 숫자형 전부 사용.
    figsize : tuple, default (14,12)
        그림 크기.
    cmap : str, default 'coolwarm'
        색상 지도.
    center : float, default 0
        히트맵 중앙값.
    vmin, vmax : float, default -1,1
        컬러스케일 축 범위.
    annot : bool, default False
        셀에 숫자 표시 여부.
    square : bool, default True
        각 셀을 정사각형으로 그릴지 여부.
    title : str, default 'Correlation Heatmap'
        그래프 제목.
    """
    # 사용할 컬럼 결정
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns

    # 상관계수 계산
    corr = df[cols].corr()

    # 플롯
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        square=square
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 풍속과의 상관관계를 구체적으로 숫자 분석
def get_wind_speed_correlations(df,
                                target='Sensor_Wind_Speed',
                                top_n=5,
                                corr_threshold=0.5):
    """
    df의 숫자형 컬럼 중 target과의 상관계수를 계산해서
     1) 양/음 방향 상위 top_n 변수
     2) 절대값 기준 corr_threshold 이상의 변수 리스트
    를 반환합니다.
    
    Returns
    -------
    corr_target : pd.Series
        target과의 상관계수 (내림차순 정렬)
    top_pos : pd.Series
        corr_target 중 양의 상관계수 상위 top_n
    top_neg : pd.Series
        corr_target 중 음의 상관계수 하위 top_n
    strong_feats : list
        |상관계수| >= corr_threshold 인 변수 이름 리스트
    """
    # 1) 숫자형 컬럼만
    num_cols = df.select_dtypes(include=[np.number]).columns
    # 2) 상관계수 행렬
    corr = df[num_cols].corr()
    # 3) 타깃과의 상관계수 시리즈
    corr_target = corr[target].drop(target).sort_values(ascending=False)
    
    # 4) 상위·하위
    top_pos = corr_target.head(top_n)
    top_neg = corr_target.tail(top_n)
    
    # 5) 절대값 기준으로 강한 연관 변수
    strong_feats = corr_target[ corr_target.abs() >= corr_threshold ].index.tolist()
    
    return corr_target, top_pos, top_neg, strong_feats

# plot_correlation_heatmap(
#     filtered,
#     annot=False,                                 # 숫자도 보고 싶다면 True
#     title='After Outlier Removal (Heatmap)'
# )

# corr_target, top_pos, top_neg, strong_feats = get_wind_speed_correlations(
#     filtered,
#     target='Sensor_Wind_Speed',
#     top_n=10,
#     corr_threshold=0.5
# )

# print("Sensor_Wind_Speed와 양의 상관 상위 5개:\n", top_pos, "\n")
# print("Sensor_Wind_Speed와 음의 상관 상위 5개:\n", top_neg, "\n")
# print("절대값 ≥ 0.5인 강한 변수들:\n", strong_feats)

# corr_target = filtered.select_dtypes(include=[np.number]).corr()['Sensor_Wind_Speed'].drop('Sensor_Wind_Speed').abs()

# for thr in [0.3, 0.5, 0.7, 0.8]:
#     n = (corr_target >= thr).sum()
#     print(f"|r| ≥ {thr: .1f} 인 변수 개수: {n}")


# 최선의 보간 방법 찾기
def compare_interpolation_methods_mae(
    df,
    reference='Sensor_Wind_Speed',
    corr_threshold=0.7,
    mask_frac=0.05,
    methods=None,
    random_state=42
):
    """
    1) df에서 reference 변수와 |corr| ≥ corr_threshold 인 숫자형 컬럼을 고르고
    2) 각 컬럼에 대해 mask_frac 비율로 인위적 NaN을 만들고
    3) 여러 보간법을 적용해 원래 값과 MAE를 계산한 뒤
    4) method × 변수 형태의 MAE 테이블을 반환합니다.
    """
    if methods is None:
        methods = {
            'linear':     lambda s: s.interpolate(method='linear'),
            'polynomial': lambda s: s.interpolate(method='polynomial', order=2),
            'spline':     lambda s: s.interpolate(method='spline', order=3),
            'ffill':      lambda s: s.ffill(),
            'bfill':      lambda s: s.bfill(),
            'time':       lambda s: s.interpolate(method='time') if isinstance(s.index, pd.DatetimeIndex) else s  # time 보간
        }

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Time interpolation을 사용하려면 DataFrame의 index가 DatetimeIndex여야 합니다.")

    # 1) strong corr 변수 선택
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()[reference].drop(reference)
    strong_cols = corr[ corr.abs() >= corr_threshold ].index.tolist()
    
    rng = np.random.RandomState(random_state)
    mae_dict = {name: {} for name in methods}
    
    # 2) 변수별 테스트
    for col in strong_cols:
        idx_valid = df.index[df[col].notna()]
        n_mask = int(len(idx_valid) * mask_frac)
        mask_idx = rng.choice(idx_valid, size=n_mask, replace=False)
        
        true_vals = df.loc[mask_idx, col]
        tmp = df.copy()
        tmp.loc[mask_idx, col] = np.nan
        
        # 3) MAE 계산
        for name, func in methods.items():
            try:
                imputed = func(tmp[col])
                mae = mean_absolute_error(true_vals, imputed.loc[mask_idx])
            except Exception:
                mae = np.nan
            mae_dict[name][col] = mae
    
    # 4) DataFrame 변환
    mae_df = pd.DataFrame(mae_dict).T
    return mae_df

# 사용 예시
# mae_df = compare_interpolation_methods_mae(filtered,
#                                            reference='Sensor_Wind_Speed',
#                                            corr_threshold=0.7,
#                                            mask_frac=0.05,
#                                            random_state=42)
# print(mae_df)


# [결과] threshold = 0.7
#             Generation_Amount  Generator_Current_A  Generator_Current_B  \
# linear               1.450813            58.402845            60.841614   
# polynomial           1.698185            65.218303            66.104166   
# spline               1.678774            67.124955            67.671468   
# ffill                1.775174            75.330452            78.503821   
# bfill                1.777074            74.287239            77.110671   
# time                 1.450100            58.433530            60.841614   

#             Generator_Current_C  Generator_Output  Generator_Speed  \
# linear                63.789608         73.944216        38.509190   
# polynomial            70.776403         84.397936        43.616722   
# spline                72.675161         86.968456        45.389639   
# ffill                 83.196696         96.376848        49.646829   
# bfill                 79.189666         95.872293        50.357992   
# time                  63.742718         73.944216        38.509190   

#             Rotor_Speed  Winding_Temperature_A  Winding_Temperature_B  \
# linear         0.319244               0.156659               0.150894   
# polynomial     0.361453               0.165780               0.149253   
# spline         0.635459               0.755524               0.754649   
# ffill          0.407928               0.230958               0.227475   
# bfill          0.411666               0.238632               0.230790   
# time           0.319244               0.156659               0.150894   

#             Winding_Temperature_C  Transformer_Temperature_A  \
# linear                   0.131961                   0.039519   
# polynomial               0.138661                   0.046185   
# spline                   0.797792                   0.733613   
# ffill                    0.207389                   0.060766   
# bfill                    0.209162                   0.057938   
# time                     0.131557                   0.039519   

#             Transformer_Temperature_B  Transformer_Temperature_C  
# linear                       0.044698                   0.045781  
# polynomial                   0.051164                   0.050526  
# spline                       0.738463                   0.785392  
# ffill                        0.068746                   0.064872  
# bfill                        0.068408                   0.062170  
# time                         0.044698                   0.045781

# time이 다 MAE가 낮다

# 실제 threshold = 0.7, time interpolation로 보간 실행
def interpolate_missing_values_time(
    df: pd.DataFrame,
    reference: str = 'Sensor_Wind_Speed',
    corr_threshold: float = 0.7
) -> pd.DataFrame:
    """
    df에서 reference 변수와 |corr| ≥ corr_threshold인 숫자형 컬럼을 선택하여,
    0을 NaN으로 간주한 뒤 'time' 보간으로 결측치를 채우고,
    앞뒤 보간(fill forward/backward)으로 남은 NaN을 최종 처리합니다.

    DatetimeIndex가 반드시 필요합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        결측 처리할 원본 데이터프레임 (극단치 제거된 상태 가정).
    reference : str, default 'Sensor_Wind_Speed'
        상관계수 기준이 될 기준 변수명.
    corr_threshold : float, default 0.7
        reference와의 |상관계수| 기준값.
    
    Returns
    -------
    pd.DataFrame
        보간 처리된 데이터프레임 (원본 복사본).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex가 필요합니다. 먼저 preprocess_datetime()을 적용하세요.")

    # 1) 숫자형 컬럼 중 상관계수가 높은 변수 선택
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()[reference].drop(reference)
    strong_cols = corr[corr.abs() >= corr_threshold].index.tolist()
    
    # 2) 보간 대상 컬럼 목록 (기준 변수 포함)
    cols_to_interp = [reference] + strong_cols

    # 3) 복사본 생성
    df_interp = df.copy()

    # 4) 각 컬럼별로 0→NaN, 'time' 보간, ffill/bfill 처리
    for col in cols_to_interp:
        df_interp[col] = df_interp[col].replace(0, np.nan)
        df_interp[col] = df_interp[col].interpolate(method='time')
        df_interp[col] = df_interp[col].bfill().ffill()

    return df_interp

