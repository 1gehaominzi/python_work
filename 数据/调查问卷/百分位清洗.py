import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv('调查问卷/青少年肥胖与饮食健康调查问卷数据.csv')

# 2. 清洗列名（规范格式）
df.columns = [col.strip().replace("(", "").replace(")", "").replace("（", "").replace("）", "")
              .replace(" ", "_").replace("，", "").replace("?", "") for col in df.columns]

# 3. 缺失值处理
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# 4. 异常值处理：百分位裁剪方法（P1 和 P99）
def cap_outliers_percentile(df, column, lower_percentile=0.01, upper_percentile=0.99):
    if df[column].dtype != 'object':
        lower_bound = df[column].quantile(lower_percentile)
        upper_bound = df[column].quantile(upper_percentile)
        df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

# 数值列（可根据你的数据列继续补充）
numerical_cols = [
    '年龄', '身高cm', '体重kg', 'BMI',
    '每周快餐摄入次数', '每日蔬菜水果摄入量份',
    '每日含糖饮料摄入量瓶', '每日饮水量升',
    '每日睡眠时间小时', '每周运动次数',
    '每次运动时长分钟', '学习压力感1-10'
]

# 应用百分位裁剪处理
for col in numerical_cols:
    col_clean = col.replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace(" ", "_").replace("，", "")
    if col_clean in df.columns:
        df = cap_outliers_percentile(df, col_clean)

# 5. 保存结果
df.to_csv('调查问卷/青少年肥胖与饮食健康调查问卷数据_百分位清洗.csv', index=False, encoding='utf-8-sig')

print("✅ 数据清洗 + 百分位异常值处理完成")
print("📁 保存为：青少年肥胖与饮食健康调查问卷数据_百分位清洗.csv")
