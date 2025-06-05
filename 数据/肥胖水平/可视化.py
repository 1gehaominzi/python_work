import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
df = pd.read_csv('肥胖水平/青少年肥胖水平处理过后.csv')

# 1. 肥胖水平分布分析
plt.figure(figsize=(12, 6))
sns.countplot(x='NObeyesdad', data=df, order=df['NObeyesdad'].value_counts().index)
plt.title('肥胖水平分布', fontsize=15)
plt.xlabel('肥胖类别')
plt.ylabel('人数')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 2. 性别与肥胖关系
plt.figure(figsize=(10, 6))
sns.countplot(x='NObeyesdad', hue='Gender', data=df)
plt.title('不同性别肥胖水平分布', fontsize=15)
plt.xlabel('肥胖类别')
plt.ylabel('人数')
plt.xticks(rotation=15)
plt.legend(title='性别')
plt.tight_layout()
plt.show()

# 3. 年龄与BMI关系
# 计算BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

plt.figure(figsize=(12, 7))
sns.scatterplot(x='Age', y='BMI', hue='NObeyesdad', data=df, palette='viridis', s=80, alpha=0.8)
plt.title('年龄与BMI关系', fontsize=15)
plt.xlabel('年龄')
plt.ylabel('BMI指数')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='肥胖类别', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. 生活习惯热力图
# 选择关键生活习惯特征
habits = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
corr = df[habits + ['BMI']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('生活习惯与BMI相关性热力图', fontsize=15)
plt.tight_layout()
plt.show()

# 5. 家族史与肥胖关系
plt.figure(figsize=(10, 6))
sns.boxplot(x='NObeyesdad', y='BMI', hue='family_history_with_overweight', data=df)
plt.title('家族肥胖史对BMI的影响', fontsize=15)
plt.xlabel('肥胖类别')
plt.ylabel('BMI指数')
plt.xticks(rotation=15)
plt.legend(title='家族肥胖史', loc='upper right')
plt.tight_layout()
plt.show()

# 6. 饮食习惯与肥胖关系
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 高热量食物摄入
sns.countplot(x='FAVC', hue='NObeyesdad', data=df, ax=axes[0, 0])
axes[0, 0].set_title('高热量食物摄入与肥胖')
axes[0, 0].set_xlabel('是否常吃高热量食物')

# 蔬菜摄入频率
sns.boxplot(x='NObeyesdad', y='FCVC', data=df, ax=axes[0, 1])
axes[0, 1].set_title('蔬菜摄入频率与肥胖')
axes[0, 1].set_ylabel('蔬菜摄入频率')
axes[0, 1].tick_params(axis='x', rotation=15)

# 两餐间零食习惯
sns.countplot(x='CAEC', hue='NObeyesdad', data=df, ax=axes[1, 0])
axes[1, 0].set_title('两餐间零食习惯与肥胖')
axes[1, 0].set_xlabel('零食习惯')

# 每日进餐次数
sns.boxplot(x='NObeyesdad', y='NCP', data=df, ax=axes[1, 1])
axes[1, 1].set_title('每日进餐次数与肥胖')
axes[1, 1].set_ylabel('每日进餐次数')
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

# 7. 身体活动与肥胖关系
plt.figure(figsize=(12, 8))
sns.boxplot(x='NObeyesdad', y='FAF', data=df)
plt.title('体育活动频率与肥胖水平', fontsize=15)
plt.xlabel('肥胖类别')
plt.ylabel('体育活动频率')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 8. 交通运输方式与肥胖关系
plt.figure(figsize=(12, 6))
transport_order = df['MTRANS'].value_counts().index
sns.countplot(x='MTRANS', hue='NObeyesdad', data=df, order=transport_order)
plt.title('交通工具使用与肥胖水平', fontsize=15)
plt.xlabel('主要交通工具')
plt.ylabel('人数')
plt.xticks(rotation=15)
plt.legend(title='肥胖类别', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()