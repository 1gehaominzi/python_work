import pandas as pd
import numpy as np
from scipy import stats

# 1. 读取数据
df = pd.read_csv('青少年肥胖与饮食健康调查问卷数据.csv')

# 2. 清洗列名（去除空格、特殊字符等）
df.columns = [col.strip().replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace(" ", "_").replace("，", "").replace("?", "") for col in df.columns]

# 3. 缺失值处理
for col in df.columns:
    if df[col].dtype == 'object':  # 分类变量
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # 数值变量
        df[col].fillna(df[col].median(), inplace=True)

# 4. 异常值处理（Z-score 方法）
def remove_outliers_zscore(df, column, threshold=3):
    if df[column].dtype != 'object':
        z_scores = np.abs(stats.zscore(df[column]))
        return df[z_scores < threshold]
    return df  # 如果是分类变量，跳过

# 要处理的数值列
numerical_cols = [
    '年龄', '身高cm', '体重kg', 'BMI',
    '每周快餐摄入次数', '每日蔬菜水果摄入量份',
    '每日含糖饮料摄入量瓶', '每日饮水量升', '每日睡眠时间小时',
    '每周运动次数', '每次运动时长分钟', '学习压力感1-10'
]

# 逐列进行 Z-score 异常值处理
for col in numerical_cols:
    col_clean = col.strip().replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace(" ", "_").replace("，", "")
    if col_clean in df.columns:
        df = remove_outliers_zscore(df, col_clean)

# 5. 保存清洗后的数据
df.to_csv('青少年肥胖与饮食健康调查问卷数据_已清洗.csv', index=False, encoding='utf-8-sig')

print("✅ 数据清洗与异常值处理完成")
print(f"📁 已保存为文件：青少年肥胖与饮食健康调查问卷数据_已清洗.csv")
print(f"🧮 剩余记录数：{df.shape[0]}")