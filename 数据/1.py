import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置可视化风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 加载数据（假设数据已保存为CSV文件）
df = pd.read_csv('数据整合_特征工程.csv')

# 1. 目标变量分布分析 (肥胖类别)
plt.figure(figsize=(10, 6))
obesity_counts = df['OBESITY'].value_counts()
ax = sns.barplot(x=obesity_counts.values, y=obesity_counts.index, palette='viridis')
plt.title('肥胖类别分布', fontsize=15)
plt.xlabel('样本数量', fontsize=12)
plt.ylabel('肥胖类别', fontsize=12)
for i, v in enumerate(obesity_counts):
    ax.text(v + 3, i, str(v), color='black', va='center')
plt.tight_layout()
plt.savefig('肥胖类别分布.png', dpi=300)

# 2. 关键连续变量分布
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 年龄分布
sns.histplot(df['Age'], kde=True, bins=20, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('年龄分布')

# BMI分布
sns.histplot(df['BMI'], kde=True, bins=25, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('BMI分布')
axes[0, 1].axvline(18.5, color='green', linestyle='--')  # 体重不足分界线
axes[0, 1].axvline(25, color='orange', linestyle='--')  # 超重分界线

# 身高分布
sns.histplot(df['Height'], kde=True, bins=20, ax=axes[1, 0], color='limegreen')
axes[1, 0].set_title('身高分布')

# 体重分布
sns.histplot(df['Weight'], kde=True, bins=25, ax=axes[1, 1], color='violet')
axes[1, 1].set_title('体重分布')

plt.tight_layout()
plt.savefig('连续变量分布.png', dpi=300)

# 3. BMI与肥胖类别的箱型图
plt.figure(figsize=(12, 8))
sns.boxplot(x='OBESITY', y='BMI', data=df, palette='Set2')
plt.title('不同肥胖类别的BMI分布', fontsize=15)
plt.xlabel('肥胖类别', fontsize=12)
plt.ylabel('BMI', fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('BMI与肥胖类别.png', dpi=300)

# 4. 性别与肥胖类别的关系
plt.figure(figsize=(10, 6))
gender_obesity = pd.crosstab(df['Gender'], df['OBESITY'])
gender_obesity.plot(kind='bar', stacked=True, colormap='tab20')
plt.title('性别与肥胖类别分布', fontsize=15)
plt.xlabel('性别', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.legend(title='肥胖类别', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('性别与肥胖类别.png', dpi=300)

# 5. 身高体重散点图（按肥胖类别）
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Height', y='Weight', hue='OBESITY', 
                data=df, palette='Set1', alpha=0.8, s=80)
plt.title('身高与体重关系（按肥胖类别）', fontsize=15)
plt.xlabel('身高 (m)', fontsize=12)
plt.ylabel('体重 (kg)', fontsize=12)
plt.legend(title='肥胖类别', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('身高体重散点图.png', dpi=300)

# 6. 关键特征相关矩阵
plt.figure(figsize=(14, 12))
features = ['Age', 'Height', 'Weight', 'BMI', 'FCVC', 'CH2O', 'FAF', 'TUE']
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
            linewidths=0.5, linecolor='lightgray')
plt.title('特征相关矩阵', fontsize=15)
plt.tight_layout()
plt.savefig('特征相关矩阵.png', dpi=300)

# 7. 年龄分组分析
df['Age_Group'] = pd.cut(df['Age'], 
                         bins=[14, 17, 20, 30], 
                         labels=['青少年 (14-17)', '青年 (18-20)', '成年人 (21-30)'])
                         
plt.figure(figsize=(10, 6))
age_group_counts = df['Age_Group'].value_counts().sort_index()
sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='muted')
plt.title('年龄分组分布', fontsize=15)
plt.xlabel('年龄分组', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.tight_layout()
plt.savefig('年龄分组分布.png', dpi=300)

# 8. 家庭肥胖史与当前体重状况的关系
plt.figure(figsize=(10, 6))
family_obesity = pd.crosstab(df['Family_history'], df['OBESITY'])
family_obesity.index = ['无家族史', '有家族史']
family_obesity.plot(kind='bar', stacked=True, colormap='tab10')
plt.title('家族肥胖史与体重状况', fontsize=15)
plt.xlabel('家族肥胖史', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.legend(title='肥胖类别', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('家族史与肥胖.png', dpi=300)

plt.show()