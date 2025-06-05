import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('肥胖预测数据收集/处理后.csv')

# 1. 数据概览
print("="*50)
print("数据概览:")
print(f"数据集形状: {df.shape}")
print("\n前5行数据:")
print(df.head())
print("\n数据类型和缺失值:")
print(df.info())
print("\n描述性统计:")
print(df.describe(include='all').T)

# 2. 肥胖水平分布分析
plt.figure(figsize=(14, 8))
obesity_order = ['Insufficient_Weight', 'Normal_Weight', 
                'Overweight_Level_I', 'Overweight_Level_II',
                'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
ax = sns.countplot(y='Obesity', data=df, order=obesity_order, palette='viridis')
plt.title('肥胖水平分布', fontsize=16)
plt.xlabel('数量', fontsize=12)
plt.ylabel('肥胖等级', fontsize=12)

# 添加数据标签
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 10, p.get_y() + p.get_height()/2, 
             f'{int(width)}', 
             ha='left', va='center')
plt.tight_layout()
plt.savefig('obesity_distribution.png', dpi=300)
plt.show()

# 3. 数值特征分析
numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'Tue']
plt.figure(figsize=(18, 12))

# 绘制分布图
for i, col in enumerate(numeric_features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'{col}分布', fontsize=12)
    plt.xlabel(col)
    plt.ylabel('频率')
plt.tight_layout()
plt.savefig('numeric_distribution.png', dpi=300)
plt.show()

# 4. BMI计算与分析
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
df['BMI_Category'] = pd.cut(df['BMI'], 
                           bins=[0, 18.5, 25, 30, 35, 40, 100],
                           labels=['体重不足', '正常', '超重I级', '超重II级', 
                                  '肥胖I级', '肥胖II/III级'])

plt.figure(figsize=(14, 8))
sns.boxplot(x='BMI_Category', y='BMI', data=df, palette='Set2')
plt.title('BMI分类分布', fontsize=16)
plt.xlabel('BMI分类')
plt.ylabel('BMI值')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('bmi_distribution.png', dpi=300)
plt.show()

# 5. 分类特征分析
categorical_features = ['Gender', 'Family_history_with_overweight', 'FAVC', 
                       'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

plt.figure(figsize=(18, 20))
for i, col in enumerate(categorical_features):
    plt.subplot(4, 2, i+1)
    sns.countplot(y=col, data=df, palette='pastel', order=df[col].value_counts().index)
    plt.title(f'{col}分布', fontsize=14)
    plt.xlabel('数量')
    plt.ylabel(col)
plt.tight_layout()
plt.savefig('categorical_distribution.png', dpi=300)
plt.show()

# 6. 肥胖水平与特征关系
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('肥胖水平与关键特征关系', fontsize=20)

# 性别与肥胖
sns.countplot(x='Obesity', hue='Gender', data=df, ax=axes[0, 0], palette='cool', order=obesity_order)
axes[0, 0].set_title('性别与肥胖水平', fontsize=14)
axes[0, 0].set_xlabel('肥胖水平')
axes[0, 0].set_ylabel('数量')
axes[0, 0].tick_params(axis='x', rotation=45)

# 家族史与肥胖
sns.countplot(x='Obesity', hue='Family_history_with_overweight', data=df, 
              ax=axes[0, 1], palette='viridis', order=obesity_order)
axes[0, 1].set_title('家族史与肥胖水平', fontsize=14)
axes[0, 1].set_xlabel('肥胖水平')
axes[0, 1].set_ylabel('数量')
axes[0, 1].tick_params(axis='x', rotation=45)

# 年龄与肥胖
sns.boxplot(x='Obesity', y='Age', data=df, ax=axes[1, 0], palette='Set3', order=obesity_order)
axes[1, 0].set_title('年龄与肥胖水平', fontsize=14)
axes[1, 0].set_xlabel('肥胖水平')
axes[1, 0].set_ylabel('年龄')
axes[1, 0].tick_params(axis='x', rotation=45)

# BMI与肥胖
sns.boxplot(x='Obesity', y='BMI', data=df, ax=axes[1, 1], palette='Set1', order=obesity_order)
axes[1, 1].set_title('BMI与肥胖水平', fontsize=14)
axes[1, 1].set_xlabel('肥胖水平')
axes[1, 1].set_ylabel('BMI')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('obesity_relationships.png', dpi=300)
plt.show()

# 7. 特征相关性分析
# 创建数值特征的相关性矩阵
corr_matrix = df[numeric_features + ['BMI']].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('数值特征相关性热力图', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# 8. 饮食习惯与肥胖关系
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('饮食习惯与肥胖水平', fontsize=20)

# 蔬菜摄入频率
sns.boxplot(x='Obesity', y='FCVC', data=df, ax=axes[0, 0], palette='YlGn', order=obesity_order)
axes[0, 0].set_title('蔬菜摄入频率与肥胖', fontsize=14)
axes[0, 0].set_xlabel('肥胖水平')
axes[0, 0].set_ylabel('蔬菜摄入频率')
axes[0, 0].tick_params(axis='x', rotation=45)

# 高热量食物摄入
sns.countplot(x='Obesity', hue='FAVC', data=df, ax=axes[0, 1], palette='OrRd', order=obesity_order)
axes[0, 1].set_title('高热量食物摄入与肥胖', fontsize=14)
axes[0, 1].set_xlabel('肥胖水平')
axes[0, 1].set_ylabel('数量')
axes[0, 1].tick_params(axis='x', rotation=45)

# 每日餐数
sns.boxplot(x='Obesity', y='NCP', data=df, ax=axes[1, 0], palette='PuBu', order=obesity_order)
axes[1, 0].set_title('每日餐数与肥胖', fontsize=14)
axes[1, 0].set_xlabel('肥胖水平')
axes[1, 0].set_ylabel('每日餐数')
axes[1, 0].tick_params(axis='x', rotation=45)

# 零食习惯
sns.countplot(x='Obesity', hue='CAEC', data=df, ax=axes[1, 1], palette='RdPu', order=obesity_order)
axes[1, 1].set_title('零食习惯与肥胖', fontsize=14)
axes[1, 1].set_xlabel('肥胖水平')
axes[1, 1].set_ylabel('数量')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].legend(title='零食习惯')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('diet_obesity.png', dpi=300)
plt.show()

# 9. 生活方式与肥胖
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('生活方式与肥胖水平', fontsize=20)

# 身体活动频率
sns.boxplot(x='Obesity', y='FAF', data=df, ax=axes[0, 0], palette='YlOrBr', order=obesity_order)
axes[0, 0].set_title('身体活动频率与肥胖', fontsize=14)
axes[0, 0].set_xlabel('肥胖水平')
axes[0, 0].set_ylabel('活动频率')
axes[0, 0].tick_params(axis='x', rotation=45)

# 屏幕使用时间
sns.boxplot(x='Obesity', y='Tue', data=df, ax=axes[0, 1], palette='Blues', order=obesity_order)
axes[0, 1].set_title('屏幕使用时间与肥胖', fontsize=14)
axes[0, 1].set_xlabel('肥胖水平')
axes[0, 1].set_ylabel('屏幕时间')
axes[0, 1].tick_params(axis='x', rotation=45)

# 饮酒习惯
sns.countplot(x='Obesity', hue='CALC', data=df, ax=axes[1, 0], palette='Greens', order=obesity_order)
axes[1, 0].set_title('饮酒习惯与肥胖', fontsize=14)
axes[1, 0].set_xlabel('肥胖水平')
axes[1, 0].set_ylabel('数量')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend(title='饮酒频率')

# 交通工具
sns.countplot(x='Obesity', hue='MTRANS', data=df, ax=axes[1, 1], palette='Purples', order=obesity_order)
axes[1, 1].set_title('交通工具与肥胖', fontsize=14)
axes[1, 1].set_xlabel('肥胖水平')
axes[1, 1].set_ylabel('数量')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].legend(title='交通工具')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('lifestyle_obesity.png', dpi=300)
plt.show()

# 10. 多维度分析：性别、年龄、BMI
plt.figure(figsize=(14, 10))
sns.scatterplot(x='Age', y='BMI', hue='Gender', size='Weight',
                sizes=(20, 200), alpha=0.7, data=df, palette='viridis')
plt.title('年龄、BMI与性别的关系', fontsize=16)
plt.xlabel('年龄')
plt.ylabel('BMI')
plt.axhline(y=18.5, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=25, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=30, color='r', linestyle='--', alpha=0.3)
plt.text(16, 17, '体重不足', fontsize=12)
plt.text(16, 21.5, '正常', fontsize=12)
plt.text(16, 27.5, '超重', fontsize=12)
plt.text(16, 35, '肥胖', fontsize=12)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('age_bmi_gender.png', dpi=300)
plt.show()