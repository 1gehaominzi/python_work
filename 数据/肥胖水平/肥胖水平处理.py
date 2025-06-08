import pandas as pd
import numpy as np

# 1. 读取源文件
df = pd.read_csv('肥胖水平/原数据.csv')

# 2. 缺失值处理
for column in df.columns:
    if df[column].dtype == 'object':  # 分类变量
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # 数值变量
        df[column].fillna(df[column].median(), inplace=True)

# 3. 异常值处理（IQR 方法）
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 处理指定连续数值列
numerical_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in numerical_cols:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)

# 4. 筛选年龄小于20岁的数据
df_under_20 = df[df['Age'] < 20]

# 5. 保存结果为新的CSV文件
df_under_20.to_csv('肥胖水平/青少年肥胖水平处理过后.csv', index=False, encoding='utf-8-sig')

print("处理完成，已保存为“年龄小于20岁.csv”。共包含记录数：", df_under_20.shape[0])