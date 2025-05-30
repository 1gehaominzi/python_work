import pandas as pd
import numpy as np
from scipy import stats

# 1. 读取数据文件
df = pd.read_csv('肥胖预测数据收集/肥胖预测数据收集.csv')

# 2. 标准化列名（可选）
df.columns = [col.strip().capitalize().replace(" ", "_") for col in df.columns]
df.rename(columns={
    'Obesity_level': 'NObesity',
    'Family_history': 'Family_history_with_overweight'
}, inplace=True)

# 3. 缺失值处理
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# 4. 异常值处理（使用 Z-score）
def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

# 应用 Z-score 异常值处理
numerical_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in numerical_cols:
    if col in df.columns:
        df = remove_outliers_zscore(df, col)

# 5. 筛选年龄小于 20 的记录
df_under_20 = df[df['Age'] < 20]

# 6. 保存结果到新的 CSV 文件
df_under_20.to_csv('肥胖预测数据收集/处理后.csv', index=False, encoding='utf-8-sig')

print("✅ 数据处理完成")
print(f"🎯 筛选出年龄小于20岁的记录共计：{df_under_20.shape[0]}")
print("📁 结果已保存为：处理后.csv")