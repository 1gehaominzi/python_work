import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据
df = pd.read_csv("数据整合_特征工程.csv")

# 设置绘图风格
sns.set(style="whitegrid", palette="pastel")
plt.figure(figsize=(12, 8))

# 1. 目标变量分布分析
plt.subplot(2, 2, 1)
obesity_order = df['OBESITY'].value_counts().index
sns.countplot(y='OBESITY', data=df, order=obesity_order)
plt.title('Obesity Category Distribution')
plt.xlabel('Count')
plt.ylabel('Obesity Level')

# 2. 年龄与BMI关系
plt.subplot(2, 2, 2)
sns.scatterplot(x='Age', y='BMI', hue='OBESITY', data=df, 
                palette='viridis', alpha=0.7)
plt.title('Age vs BMI by Obesity Category')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. 性别与肥胖类型关系
plt.subplot(2, 2, 3)
gender_obesity = pd.crosstab(df['Gender'], df['OBESITY'])
gender_obesity.plot(kind='bar', stacked=True)
plt.title('Gender Distribution Across Obesity Levels')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 4. 特征相关性热力图（选择关键特征）
plt.subplot(2, 2, 4)
selected_features = ['Age', 'Height', 'Weight', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
corr_matrix = df[selected_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')

plt.tight_layout()
plt.show()

# 5. BMI分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(df['BMI'], bins=30, kde=True)
plt.axvline(x=18.5, color='r', linestyle='--', label='Underweight')
plt.axvline(x=25, color='g', linestyle='--', label='Healthy')
plt.axvline(x=30, color='b', linestyle='--', label='Overweight')
plt.title('BMI Distribution with Classification Boundaries')
plt.legend()
plt.show()

# 6. 每日饮水量与肥胖类型关系
plt.figure(figsize=(10, 6))
sns.boxplot(x='OBESITY', y='CH2O', data=df, order=obesity_order)
plt.title('Daily Water Consumption by Obesity Category')
plt.xticks(rotation=45)
plt.ylabel('Liters of Water Daily')
plt.show()

# 7. 体力活动频率分布
plt.figure(figsize=(10, 6))
sns.countplot(x='FAF', data=df, hue='Gender')
plt.title('Physical Activity Frequency Distribution by Gender')
plt.xlabel('Physical Activity Frequency (days/week)')
plt.ylabel('Count')
plt.show()

# 8. 年龄组与肥胖类型关系
age_groups = ['AgeGrp_Teen', 'AgeGrp_Young_Adult']
age_obesity = df.groupby('OBESITY')[age_groups].mean()
age_obesity.plot(kind='bar', stacked=True)
plt.title('Age Group Distribution Across Obesity Levels')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.show()

# 9. 特征重要性分析（基于随机森林）
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 准备数据
X = df.drop(['OBESITY', 'Gender'], axis=1)
y = LabelEncoder().fit_transform(df['OBESITY'])

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 获取特征重要性
importance = rf.feature_importances_
features = X.columns
indices = np.argsort(importance)[-15:]  # 取前15个重要特征

# 绘制特征重要性
plt.figure(figsize=(12, 8))
plt.title('Top 15 Important Features')
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()