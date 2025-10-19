# -*- coding: utf-8 -*-
""" remove_outlier.py
극단 처리 전처리 함수들
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# raw한 data 원본
def plot_generator_features(df, scatter_area=5, columns_per_row=5):
    """
    발전기 센서 데이터를 효과적으로 시각화하는 함수.
    
    - _A/_B/_C로 끝나는 센서는 같은 subplot에 묶어서 그림.
    - 나머지는 각기 다른 subplot에 독립적으로 그림.

    Parameters:
    - df: DataFrame (예: generator_bottom)
    - scatter_area: scatter plot의 점 크기
    - columns_per_row: subplot 당 열 개수 (기본값 5)
    """
    data_size = len(df)
    used = set()
    plots = []

    # 그룹핑된 컬럼 정리
    for col in df.columns:
        if col in used or df[col].dtype == object:
            continue

        # A/B/C 그룹 찾기
        match = re.match(r'(.+)_A$', col)
        if match:
            base = match.group(1)
            cols = [f"{base}_{suffix}" for suffix in ['A', 'B', 'C'] if f"{base}_{suffix}" in df.columns]
            used.update(cols)
            plots.append((cols, f"{base.replace('_', ' ').title()} ABC"))
        elif not col.endswith(('_B', '_C')):  # 이미 ABC로 묶인 것 제외
            used.add(col)
            plots.append(([col], col.replace('_', ' ').title()))

    # subplot 설정
    total = len(plots)
    rows = (total + columns_per_row - 1) // columns_per_row
    fig, ax = plt.subplots(rows, columns_per_row, figsize=(columns_per_row * 4, rows * 3))
    ax = ax.flatten()

    # 데이터 시각화
    for i, (cols, title) in enumerate(plots):
        for j, col in enumerate(cols):
            color = ['r', 'g', 'b'][j % 3] if len(cols) > 1 else 'r'
            ax[i].scatter(df.index, df[col], s=scatter_area if len(cols) == 1 else 1,
                        c=color, alpha=0.5, label=col)
        ax[i].set_title(title)
        if len(cols) > 1:
            ax[i].legend(fontsize='small')


    # 빈 subplot 숨기기
    for i in range(len(plots), len(ax)):
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()

# raw한 data 원본(boxplot)
def boxplot_generator_features(df, columns_per_row=5):
    """
    발전기 센서 데이터를 boxplot으로 시각화하는 함수.
    
    - _A/_B/_C로 끝나는 센서는 같은 subplot에 묶어서 그림.
    - 나머지는 각기 다른 subplot에 독립적으로 그림.

    Parameters:
    - df: DataFrame (예: generator_bottom)
    - columns_per_row: subplot 당 열 개수 (기본값: 5)
    """
    used = set()
    plots = []

    # 그룹핑된 컬럼 정리
    for col in df.columns:
        if col in used or df[col].dtype == object:
            continue

        # A/B/C 그룹 묶기
        match = re.match(r'(.+)_A$', col)
        if match:
            base = match.group(1)
            cols = [f"{base}_{suffix}" for suffix in ['A', 'B', 'C'] if f"{base}_{suffix}" in df.columns]
            used.update(cols)
            plots.append((cols, f"{base.replace('_', ' ').title()} ABC"))
        elif not col.endswith(('_B', '_C')):
            used.add(col)
            plots.append(([col], col.replace('_', ' ').title()))

    # subplot 설정
    total = len(plots)
    rows = (total + columns_per_row - 1) // columns_per_row
    fig, ax = plt.subplots(rows, columns_per_row, figsize=(columns_per_row * 4, rows * 3))
    ax = ax.flatten()

    # Boxplot 시각화
    for i, (cols, title) in enumerate(plots):
        data = [df[col].dropna() for col in cols]
        ax[i].boxplot(data, labels=cols, patch_artist=True)
        ax[i].set_title(title, fontsize=10)
        ax[i].tick_params(axis='x', rotation=45)

    # 빈 subplot 숨기기
    for i in range(len(plots), len(ax)):
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()

# raw한 data 원본에 극단을 빼기위한 선 긋기
def plot_generator_features_exclude_outliers(df, scatter_area=5, columns_per_row=5):
    """
    발전기 센서 데이터를 효과적으로 시각화하는 함수.
    
    - _A/_B/_C로 끝나는 센서는 같은 subplot에 묶어서 그림.
    - 나머지는 각기 다른 subplot에 독립적으로 그림.
    - 특정 변수에는 기준선(horizonal line)을 그려준다.
    """
    data_size = len(df)
    used = set()
    plots = []

    # 1) 어떤 plot에 선을 그릴지 미리 정의
    ref_lines = {
        'Generation Amount':               50,
        'Coolant Temperature':             20,
        'Transformer Temperature Bus':     27,
        'Transformer Temperature ABC':     37,
        'Generator Voltage ABC':           350,
        'Pitch Pressure':                  150,
    }

    # 2) 그룹핑된 컬럼 정리
    for col in df.columns:
        if col in used or df[col].dtype == object:
            continue

        m = re.match(r'(.+)_A$', col)
        if m:
            base = m.group(1)
            cols = [f"{base}_{s}" for s in ['A','B','C'] if f"{base}_{s}" in df.columns]
            used.update(cols)
            title = f"{base.replace('_', ' ').title()} ABC"
            plots.append((cols, title))
        elif not col.endswith(('_B','_C')):
            used.add(col)
            title = col.replace('_', ' ').title()
            plots.append(([col], title))

    # 3) subplot 설정
    total = len(plots)
    rows = (total + columns_per_row - 1) // columns_per_row
    fig, axes = plt.subplots(rows, columns_per_row, figsize=(columns_per_row*4, rows*3))
    axes = axes.flatten()

    # 4) 데이터 시각화
    for i, (cols, title) in enumerate(plots):
        ax = axes[i]
        for j, col in enumerate(cols):
            color = ['r','g','b'][j] if len(cols)>1 else 'r'
            size  = scatter_area if len(cols)==1 else 1
            ax.scatter(np.arange(data_size), df[col], s=size, c=color, alpha=0.5, label=col)
        ax.set_title(title, fontsize=10)
        if len(cols)>1:
            ax.legend(fontsize='small')

        # 5) 기준선이 필요한 plot이면 그려주기
        if title in ref_lines:
            y0 = ref_lines[title]
            ax.axhline(y=y0, color='k', linestyle='--', linewidth=1)
            # optional: ax.text(…) 로 선 옆에 값 표시도 가능

    # 6) 남은 빈 subplot 숨기기
    for j in range(len(plots), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# 실제로 제외하는 함수
def filter_generator_outliers(df):
    """
    발전기 데이터에서 도메인 지식 기준으로 극단 스파이크(정해진 임계치 밖) 행을 제거합니다.
    
    - Generation_Amount          : 값 >= 50 은 제거 → keep if < 50
    - Coolant_Temperature        : 값 <= 20 은 제거 → keep if > 20
    - Transformer_Temperature_BUS: 값 <= 27 and >= 43은 제거 → keep if > 27 and < 43
    - Transformer_Temperature_ABC: A/B/C 모두 <= 37 은 제거 → keep if all > 37
    - Generator_Voltage_ABC      : A/B/C 모두 <= 350 은 제거 → keep if all > 350
    - Pitch_Pressure             : 값 <= 150 은 제거 → keep if > 150
    """
    df2 = df.copy()
    
    # 1) Generation_Amount < 50
    df2 = df2[df2['Generation_Amount'] < 50]
    
    # 2) Coolant_Temperature > 20
    df2 = df2[df2['Coolant_Temperature'] > 20]
    
    # 3) Transformer_Temperature_BUS > 27
    df2 = df2[df2['Transformer_Temperature_BUS'] > 27]
    df2 = df2[df2['Transformer_Temperature_BUS'] < 43]
    
    # 4) Transformer_Temperature_A/B/C 모두 > 37
    tcols = ['Transformer_Temperature_A',
             'Transformer_Temperature_B',
             'Transformer_Temperature_C']
    df2 = df2[(df2[tcols] > 37).all(axis=1)]
    
    # 5) Generator_Voltage_A/B/C 모두 > 350
    vcols = ['Generator_Voltage_A',
             'Generator_Voltage_B',
             'Generator_Voltage_C']
    df2 = df2[(df2[vcols] > 350).all(axis=1)]
    
    # 6) Pitch_Pressure > 150
    df2 = df2[df2['Pitch_Pressure'] > 150]
    
    return df2

