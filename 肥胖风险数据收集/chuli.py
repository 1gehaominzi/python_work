import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv('肥胖风险数据收集/obesity_level.csv')

# 2. 重命名标签（可选，根据你上面列出的字段修正可能的列名）
df.columns = [col.strip().capitalize().replace(" ", "_") for col in df.columns]
df.rename(columns={
    '0be1dad': 'NObeyesdad'
}, inplace=True)

# 3. 缺失值处理
for column in df.columns:
    if df[column].dtype == 'object':  # 类别变量
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # 数值变量
        df[column].fillna(df[column].median(), inplace=True)

# 4. 异常值处理（IQR 方法）
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 数值型列（连续变量）
numerical_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in numerical_cols:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)

# 5. 筛选年龄小于20岁
df_under_20 = df[df['Age'] < 20]

# 6. 保存为新的CSV文件
df_under_20.to_csv('肥胖风险数据收集/处理后.csv', index=False, encoding='utf-8-sig')

print("数据清洗完成，共计：", df_under_20.shape[0])
print("已保存为：处理后.csv")