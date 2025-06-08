import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 读取数据
df = pd.read_csv('调查问卷/青少年肥胖与饮食健康调查问卷数据_已清洗.csv')

# 1. 肥胖比例分析
plt.figure(figsize=(10, 6))
obesity_counts = df['是否肥胖'].value_counts()
plt.pie(obesity_counts, labels=obesity_counts.index, autopct='%1.1f%%',
        colors=['#66b3ff', '#ff9999'], startangle=90)
plt.title('青少年肥胖比例分布')
plt.tight_layout()
plt.savefig('obesity_proportion.png')
plt.show()

# 2. 肥胖与性别的关系
plt.figure(figsize=(10, 6))
gender_obesity = pd.crosstab(df['性别'], df['是否肥胖'])
gender_obesity.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999'])
plt.title('不同性别肥胖比例分布')
plt.xlabel('性别')
plt.ylabel('人数')
plt.xticks(rotation=0)
plt.legend(title='是否肥胖')
plt.tight_layout()
plt.savefig('gender_obesity.png')
plt.show()

# 3. 肥胖与年龄的关系
plt.figure(figsize=(12, 6))
sns.boxplot(x='是否肥胖', y='年龄', data=df, palette=['#66b3ff', '#ff9999'])
plt.title('肥胖与非肥胖群体年龄分布对比')
plt.xlabel('是否肥胖')
plt.ylabel('年龄')
plt.tight_layout()
plt.savefig('age_obesity.png')
plt.show()

# 4. 饮食习惯与肥胖的关系
diet_factors = ['每日三餐是否规律', '每周快餐摄入次数', '每日蔬菜水果摄入量份', '每日含糖饮料摄入量瓶']
plt.figure(figsize=(16, 12))

for i, factor in enumerate(diet_factors, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='是否肥胖', y=factor, data=df, palette=['#66b3ff', '#ff9999'])
    plt.title(f'肥胖 vs {factor}')
    plt.xlabel('是否肥胖')

plt.tight_layout()
plt.savefig('diet_factors.png')
plt.show()

# 5. 运动习惯与肥胖的关系
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='是否肥胖', y='每周运动次数', data=df, palette=['#66b3ff', '#ff9999'])
plt.title('每周运动次数对比')

plt.subplot(1, 2, 2)
sns.boxplot(x='是否肥胖', y='每次运动时长分钟', data=df, palette=['#66b3ff', '#ff9999'])
plt.title('每次运动时长对比')

plt.tight_layout()
plt.savefig('exercise_factors.png')
plt.show()

# 6. 睡眠与肥胖的关系
plt.figure(figsize=(10, 6))
sns.boxplot(x='是否肥胖', y='每日睡眠时间小时', data=df, palette=['#66b3ff', '#ff9999'])
plt.title('每日睡眠时间对比')
plt.tight_layout()
plt.savefig('sleep_obesity.png')
plt.show()

# 7. 家族肥胖史与肥胖的关系
plt.figure(figsize=(10, 6))
family_obesity = pd.crosstab(df['是否有家族肥胖史'], df['是否肥胖'])
family_obesity.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999'])
plt.title('家族肥胖史与肥胖关系')
plt.xlabel('是否有家族肥胖史')
plt.ylabel('人数')
plt.xticks(rotation=0)
plt.legend(title='是否肥胖')
plt.tight_layout()
plt.savefig('family_obesity.png')
plt.show()

# 8. 学习压力与肥胖的关系
plt.figure(figsize=(12, 6))
sns.boxplot(x='是否肥胖', y='学习压力感1-10', data=df, palette=['#66b3ff', '#ff9999'])
plt.title('学习压力感对比')
plt.xlabel('是否肥胖')
plt.ylabel('学习压力感 (1-10分)')
plt.tight_layout()
plt.savefig('stress_obesity.png')
plt.show()

# 9. 肥胖相关因素热力图
# 选择数值型列
numeric_cols = ['年龄', '身高cm', '体重kg', 'BMI', '每周快餐摄入次数', 
               '每日蔬菜水果摄入量份', '每日含糖饮料摄入量瓶', '每日饮水量升',
               '每日睡眠时间小时', '每周运动次数', '每次运动时长分钟', '学习压力感1-10']

plt.figure(figsize=(14, 10))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('肥胖相关因素相关系数热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# 10. BMI分布直方图
plt.figure(figsize=(12, 6))
sns.histplot(df['BMI'], bins=30, kde=True, color='skyblue')
plt.axvline(x=24, color='red', linestyle='--', label='超重阈值 (BMI=24)')
plt.axvline(x=28, color='darkred', linestyle='--', label='肥胖阈值 (BMI=28)')
plt.title('BMI分布直方图')
plt.xlabel('BMI指数')
plt.ylabel('人数')
plt.legend()
plt.tight_layout()
plt.savefig('bmi_distribution.png')
plt.show()