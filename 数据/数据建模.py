import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载与初步探索
df = pd.read_csv('combined_obesity_data.csv')
print("数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())
print("\n数据类型和缺失值:")
print(df.info())

# 2. 数据预处理
# 删除全为空值的列（SMOKE, SCC, TUE）
df = df.dropna(axis=1, how='all')
print("\n删除空列后的形状:", df.shape)

# 处理剩余缺失值（如果有）
print("\n缺失值统计:")
print(df.isnull().sum())

# 编码分类特征
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CALC', 'CAEC', 'MTRANS', 'Obesity_Category']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"\n{col}编码映射:")
        print(dict(zip(le.classes_, le.transform(le.classes_))))

# 3. 特征工程与选择
# 删除可能造成数据泄露的BMI列（因为肥胖类别基于BMI计算）
if 'BMI' in df.columns:
    df = df.drop('BMI', axis=1)
    
# 特征/目标分离
X = df.drop('Obesity_Category', axis=1)
y = df['Obesity_Category']

# 4. 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)

# 5. 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 模型训练（使用随机森林）
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 7. 模型评估
y_pred = model.predict(X_test_scaled)

print("\n模型准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=label_encoders['Obesity_Category'].classes_))

# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=label_encoders['Obesity_Category'].classes_,
            yticklabels=label_encoders['Obesity_Category'].classes_)
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# 8. 特征重要性分析
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.sort_values().plot(kind='barh')
plt.title('特征重要性')
plt.show()

# 9. 模型解释（SHAP值 - 可选）
# 需要安装shap库: pip install shap
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_scaled)
    
    plt.title('特征影响力摘要')
    shap.summary_plot(shap_values, X_train, feature_names=X.columns, 
                     class_names=label_encoders['Obesity_Category'].classes_)
except ImportError:
    print("SHAP库未安装，跳过解释性分析")

# 10. 模型保存（可选）
import joblib
joblib.dump(model, 'obesity_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders['Obesity_Category'], 'label_encoder.pkl')