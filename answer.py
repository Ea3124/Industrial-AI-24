import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 1. 파일 불러오기
gt_pub = pd.read_csv('./answer/public_answer.csv', encoding='utf-8-sig')      # 실제 정답
pred_pub = pd.read_csv('./submission_public.csv', encoding='utf-8-sig')  # 예측 결과
gt_pri = pd.read_csv('./answer/private_answer.csv', encoding='utf-8-sig')      # 실제 정답
pred_pri = pd.read_csv('./submission_private.csv', encoding='utf-8-sig')  # 예측 결과

gt_pub['시간'] = pd.to_datetime(gt_pub['시간'])
pred_pub['시간'] = pd.to_datetime(pred_pub['시간'])

gt_pri['시간'] = pd.to_datetime(gt_pub['시간'])
pred_pri['시간'] = pd.to_datetime(pred_pub['시간'])

# 2. 시간 기준 병합
merged = pd.merge(gt_pub, pred_pub, on='시간', suffixes=('_true', '_pred'))
merged_pri = pd.merge(gt_pri, pred_pri, on='시간', suffixes=('_true', '_pred'))

# print(merged.columns)

# 3. 정확도 지표 계산
mae = mean_absolute_error(merged['발전량_true'], merged['발전량_pred'])
mae_pri = mean_absolute_error(merged_pri['발전량_true'], merged_pri['발전량_pred'])

print(f"📊 public MAE:  {mae:.4f}")
print(f"📊 private MAE:  {mae_pri:.4f}")

# 4. public 시각화
plt.rcParams['font.family'] = 'AppleGothic'  # 또는 원하는 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 방지

plt.figure(figsize=(12, 5))
plt.plot(merged_pri['시간'], merged_pri['발전량_true'], label='Actual', marker='o')
plt.plot(merged_pri['시간'], merged_pri['발전량_pred'], label='Predicted', marker='x')
plt.xlabel('시간')
plt.ylabel('Generation')
plt.title('Predicted vs Actual Generation (Private)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
