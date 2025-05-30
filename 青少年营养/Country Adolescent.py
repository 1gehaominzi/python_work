import pandas as pd
import numpy as np
import chardet

# 1. 检测文件编码
file_path = '青少年营养/Country Adolescent.csv'
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
file_encoding = result['encoding']
print(f"检测到的文件编码: {file_encoding}")

# 2. 使用正确编码加载数据
df = pd.read_csv(file_path, encoding=file_encoding)

# 3. 初步数据检查
print("初始形状:", df.shape)
print("缺失值统计:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# 4. 识别年份列
year_cols = [col for col in df.columns if any(str(year) in col for year in range(2000,2020))]
static_cols = [col for col in df.columns if col not in year_cols]
print(f"识别出 {len(year_cols)} 个年份列")

# 5. 分离实际数据和预测数据
actual_df = df[~df['disagg.value'].str.contains('Projection', na=False)].copy()
projection_df = df[df['disagg.value'].str.contains('Projection', na=False)].copy()

print(f"实际数据行数: {len(actual_df)}")
print(f"预测数据行数: {len(projection_df)}")

# 6. 缺失值处理 - 仅针对年份列
def fill_missing(group):
    # 仅处理年份列
    year_data = group[year_cols]
    
    # 先尝试前后填充
    filled = year_data.ffill(axis=1).bfill(axis=1)
    
    # 如果仍有缺失，使用中位数填充
    if filled.isnull().any().any():
        for col in year_cols:
            if filled[col].isnull().any():
                median_val = filled[col].median()
                filled[col] = filled[col].fillna(median_val)
    
    # 将填充后的数据合并回原始分组
    group[year_cols] = filled
    return group

# 按国家+性别分组填充
actual_df = actual_df.groupby(['iso3', 'disagg.value']).apply(fill_missing).reset_index(drop=True)

# 7. 异常值处理
## 7.1 定义合理范围 (0-100%)
def cap_outliers(series):
    return np.clip(series, 0, 100)

## 7.2 应用异常值修正
for col in year_cols:
    actual_df[col] = actual_df[col].apply(cap_outliers)

# 8. 数据验证
print("\n处理后缺失值统计:")
print(actual_df[year_cols].isnull().sum().sum())  # 应为0

print("\n异常值检查:")
print("最小值:", actual_df[year_cols].min().min())
print("最大值:", actual_df[year_cols].max().max())

# 9. 保存处理结果
actual_df.to_csv('青少年营养/Cleaned_Country_Adolescent.csv', index=False, encoding='utf-8-sig')
projection_df.to_csv('青少年营养/Projection_Data.csv', index=False, encoding='utf-8-sig')

print("\n预处理完成! 保存了实际数据和预测数据文件")
print(f"实际数据形状: {actual_df.shape}")
print(f"预测数据形状: {projection_df.shape}")