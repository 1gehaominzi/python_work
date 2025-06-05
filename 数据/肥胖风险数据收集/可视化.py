import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取数据
df = pd.read_csv('肥胖风险数据收集/处理后.csv')

# 数据初步分析
print(df.describe())
print(df.info())

# 筛选数值列
numeric_cols = ['Age', 'Height', 'Weight', 'Fcvc', 'Ncp', 'Ch2o', 'Faf', 'Tue']
df_numeric = df[numeric_cols]

# 计算相关系数矩阵
corr_matrix = df_numeric.corr()

# 可视化相关系数矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Columns')
plt.savefig('correlation_matrix_numeric.png')

# 可视化分类变量与目标变量的关系
for col in ['Gender', 'Family_history_with_overweight', 'Favc', 'Caec', 'Smoke', 'Scc', 'Mtrans']:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, hue='NObeyesdad')
    plt.title(f'{col} vs Obesity Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{col}_vs_obesity_level.png')

# 可视化数值变量与目标变量的关系
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='NObeyesdad', y=col)
    plt.title(f'{col} vs Obesity Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{col}_vs_obesity_level.png')

# 可视化数值变量的分布
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'distribution_{col}.png')