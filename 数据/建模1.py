# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# # from sklearn.svm import SVC
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# # from sklearn.model_selection import cross_val_score
# # from xgboost import XGBClassifier
# # import time

# # # 设置随机种子确保可复现性
# # np.random.seed(42)

# # # 1. 加载数据
# # # 注意：这里使用您提供的示例数据，实际应用中应替换为文件路径
# # # df = pd.read_csv('combined_obesity_data.csv')
# # # 为演示目的，我们创建模拟数据（实际应使用真实数据）
# # df = pd.read_csv('combined_obesity_data.csv')
# # print(f"数据集形状: {df.shape}")
# # print("\n前5行数据:")
# # print(df.head())
# # print("\n肥胖类别分布:")
# # print(df['Obesity_Category'].value_counts())

# # # 2. 数据预处理
# # # 编码分类特征
# # categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CALC', 'CAEC', 'MTRANS']
# # label_encoders = {}

# # for col in categorical_cols:
# #     le = LabelEncoder()
# #     df[col] = le.fit_transform(df[col])
# #     label_encoders[col] = le

# # # 编码目标变量
# # le_target = LabelEncoder()
# # df['Obesity_Category_encoded'] = le_target.fit_transform(df['Obesity_Category'])

# # # 特征/目标分离
# # X = df.drop(['Obesity_Category', 'Obesity_Category_encoded'], axis=1)
# # y = df['Obesity_Category_encoded']

# # # 数据分割
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, random_state=42, stratify=y
# # )

# # # 特征缩放
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # 3. 模型定义和训练
# # models = {
# #     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
# #     'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
# #     'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
# #     'KNN': KNeighborsClassifier(n_neighbors=5),
# #     'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
# #     'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss'),
# #     'Naive Bayes': GaussianNB()
# # }

# # # 存储结果
# # results = {
# #     'Model': [],
# #     'Accuracy': [],
# #     'Training Time (s)': [],
# #     'Prediction Time (s)': [],
# #     'Cross-Val Score': []
# # }

# # # 训练和评估模型
# # for name, model in models.items():
# #     print(f"\n{'='*50}")
# #     print(f"训练模型: {name}")
# #     start_time = time.time()
    
# #     # 训练
# #     model.fit(X_train_scaled, y_train)
# #     train_time = time.time() - start_time
    
# #     # 预测
# #     start_pred = time.time()
# #     y_pred = model.predict(X_test_scaled)
# #     pred_time = time.time() - start_pred
    
# #     # 评估
# #     accuracy = accuracy_score(y_test, y_pred)
# #     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
# #     cv_mean = np.mean(cv_scores)
    
# #     # 存储结果
# #     results['Model'].append(name)
# #     results['Accuracy'].append(accuracy)
# #     results['Training Time (s)'].append(train_time)
# #     results['Prediction Time (s)'].append(pred_time)
# #     results['Cross-Val Score'].append(cv_mean)
    
# #     # 打印分类报告
# #     print(f"\n{name} 准确率: {accuracy:.4f}")
# #     print(f"交叉验证平均准确率: {cv_mean:.4f}")
# #     print(f"训练时间: {train_time:.4f}秒")
# #     print(f"预测时间: {pred_time:.6f}秒")
    
# #     # 打印详细分类报告
# #     print(f"\n{name} 分类报告:")
# #     print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
# #     # 绘制混淆矩阵
# #     plt.figure(figsize=(10, 8))
# #     cm = confusion_matrix(y_test, y_pred)
# #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# #                 xticklabels=le_target.classes_,
# #                 yticklabels=le_target.classes_)
# #     plt.title(f'{name} 混淆矩阵')
# #     plt.ylabel('真实标签')
# #     plt.xlabel('预测标签')
# #     plt.tight_layout()
# #     plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300)
# #     plt.close()

# # # 4. 结果比较
# # results_df = pd.DataFrame(results)
# # print("\n" + "="*50)
# # print("模型性能比较:")
# # print(results_df.sort_values(by='Accuracy', ascending=False))

# # # 可视化比较
# # plt.figure(figsize=(14, 8))

# # # 准确率比较
# # plt.subplot(2, 2, 1)
# # sns.barplot(x='Accuracy', y='Model', data=results_df.sort_values('Accuracy', ascending=False))
# # plt.title('模型准确率比较')
# # plt.xlim(0.7, 1.0)

# # # 交叉验证准确率比较
# # plt.subplot(2, 2, 2)
# # sns.barplot(x='Cross-Val Score', y='Model', data=results_df.sort_values('Cross-Val Score', ascending=False))
# # plt.title('交叉验证准确率比较')
# # plt.xlim(0.7, 1.0)

# # # 训练时间比较
# # plt.subplot(2, 2, 3)
# # sns.barplot(x='Training Time (s)', y='Model', data=results_df.sort_values('Training Time (s)'))
# # plt.title('训练时间比较 (秒)')

# # # 预测时间比较
# # plt.subplot(2, 2, 4)
# # sns.barplot(x='Prediction Time (s)', y='Model', data=results_df.sort_values('Prediction Time (s)'))
# # plt.title('预测时间比较 (秒)')

# # plt.tight_layout()
# # plt.savefig('model_comparison.png', dpi=300)
# # plt.show()

# # # 5. 特征重要性分析 (针对树模型)
# # plt.figure(figsize=(12, 8))
# # for name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
# #     model = models[name]
    
# #     if hasattr(model, 'feature_importances_'):
# #         importances = model.feature_importances_
# #         indices = np.argsort(importances)[::-1]
        
# #         plt.figure()
# #         plt.title(f"{name} - 特征重要性")
# #         plt.bar(range(X_train.shape[1]), importances[indices], align='center')
# #         plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
# #         plt.tight_layout()
# #         plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png', dpi=300)
# #         plt.close()

# # # 6. 最佳模型保存
# # best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
# # best_model = models[best_model_name]
# # print(f"\n最佳模型: {best_model_name}")

# # # 保存模型
# # import joblib
# # joblib.dump(best_model, 'best_obesity_model.pkl')
# # joblib.dump(scaler, 'scaler.pkl')
# # joblib.dump(le_target, 'label_encoder.pkl')
# # print("模型保存完成!")



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.model_selection import cross_val_score
# from xgboost import XGBClassifier
# import time
# import joblib

# # 设置随机种子确保可复现性
# np.random.seed(42)

# # 1. 加载数据
# # 替换为您的实际文件路径
# df = pd.read_csv('combined_obesity_data.csv')
# print(f"数据集形状: {df.shape}")
# print("\n前5行数据:")
# print(df.head())
# print("\n肥胖类别分布:")
# print(df['Obesity_Category'].value_counts())

# # 2. 数据预处理 - 修正部分
# # 识别非数值列
# non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
# print("\n非数值列:", non_numeric_cols)

# # 移除可能导致问题的空列
# df = df.dropna(axis=1, how='all')

# # 编码分类特征
# categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CALC', 'CAEC', 'MTRANS']
# label_encoders = {}

# for col in categorical_cols:
#     if col in df.columns:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))  # 确保所有值都是字符串
#         label_encoders[col] = le
#         print(f"\n{col}编码映射:")
#         print(dict(zip(le.classes_, le.transform(le.classes_))))

# # 编码目标变量
# le_target = LabelEncoder()
# df['Obesity_Category_encoded'] = le_target.fit_transform(df['Obesity_Category'])

# # 特征/目标分离
# X = df.drop(['Obesity_Category', 'Obesity_Category_encoded'], axis=1)
# y = df['Obesity_Category_encoded']

# # 确保所有特征都是数值类型
# print("\n特征数据类型:")
# print(X.dtypes)

# # 数据分割
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# print("\n训练集形状:", X_train.shape)
# print("测试集形状:", X_test.shape)

# # 特征缩放 - 只对数值列进行缩放
# # 首先识别数值列
# numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
# print("\n数值列:", numeric_cols)

# # 仅对数值列进行缩放
# scaler = StandardScaler()
# X_train_scaled = X_train.copy()
# X_test_scaled = X_test.copy()

# X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
# X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# # 3. 模型定义和训练
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#     'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
#     'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
#     'KNN': KNeighborsClassifier(n_neighbors=5),
#     'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
#     'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss'),
#     'Naive Bayes': GaussianNB()
# }

# # 存储结果
# results = {
#     'Model': [],
#     'Accuracy': [],
#     'Training Time (s)': [],
#     'Prediction Time (s)': [],
#     'Cross-Val Score': []
# }

# # 训练和评估模型
# for name, model in models.items():
#     print(f"\n{'='*50}")
#     print(f"训练模型: {name}")
#     start_time = time.time()
    
#     # 训练
#     model.fit(X_train_scaled, y_train)
#     train_time = time.time() - start_time
    
#     # 预测
#     start_pred = time.time()
#     y_pred = model.predict(X_test_scaled)
#     pred_time = time.time() - start_pred
    
#     # 评估
#     accuracy = accuracy_score(y_test, y_pred)
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
#     cv_mean = np.mean(cv_scores)
    
#     # 存储结果
#     results['Model'].append(name)
#     results['Accuracy'].append(accuracy)
#     results['Training Time (s)'].append(train_time)
#     results['Prediction Time (s)'].append(pred_time)
#     results['Cross-Val Score'].append(cv_mean)
    
#     # 打印分类报告
#     print(f"\n{name} 准确率: {accuracy:.4f}")
#     print(f"交叉验证平均准确率: {cv_mean:.4f}")
#     print(f"训练时间: {train_time:.4f}秒")
#     print(f"预测时间: {pred_time:.6f}秒")
    
#     # 打印详细分类报告
#     print(f"\n{name} 分类报告:")
#     print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
#     # 绘制混淆矩阵
#     plt.figure(figsize=(10, 8))
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=le_target.classes_,
#                 yticklabels=le_target.classes_)
#     plt.title(f'{name} 混淆矩阵')
#     plt.ylabel('真实标签')
#     plt.xlabel('预测标签')
#     plt.tight_layout()
#     plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300)
#     plt.close()

# # 4. 结果比较
# results_df = pd.DataFrame(results)
# print("\n" + "="*50)
# print("模型性能比较:")
# print(results_df.sort_values(by='Accuracy', ascending=False))

# # 可视化比较
# plt.figure(figsize=(14, 8))

# # 准确率比较
# plt.subplot(2, 2, 1)
# sns.barplot(x='Accuracy', y='Model', data=results_df.sort_values('Accuracy', ascending=False))
# plt.title('模型准确率比较')
# plt.xlim(0.7, 1.0)

# # 交叉验证准确率比较
# plt.subplot(2, 2, 2)
# sns.barplot(x='Cross-Val Score', y='Model', data=results_df.sort_values('Cross-Val Score', ascending=False))
# plt.title('交叉验证准确率比较')
# plt.xlim(0.7, 1.0)

# # 训练时间比较
# plt.subplot(2, 2, 3)
# sns.barplot(x='Training Time (s)', y='Model', data=results_df.sort_values('Training Time (s)'))
# plt.title('训练时间比较 (秒)')

# # 预测时间比较
# plt.subplot(2, 2, 4)
# sns.barplot(x='Prediction Time (s)', y='Model', data=results_df.sort_values('Prediction Time (s)'))
# plt.title('预测时间比较 (秒)')

# plt.tight_layout()
# plt.savefig('model_comparison.png', dpi=300)
# plt.show()

# # 5. 特征重要性分析 (针对树模型)
# plt.figure(figsize=(12, 8))
# for name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
#     model = models[name]
    
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#         indices = np.argsort(importances)[::-1]
        
#         plt.figure(figsize=(10, 6))
#         plt.title(f"{name} - 特征重要性")
#         plt.bar(range(X_train.shape[1]), importances[indices], align='center')
#         plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
#         plt.tight_layout()
#         plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png', dpi=300)
#         plt.close()

# # 6. 最佳模型保存
# best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
# best_model = models[best_model_name]
# print(f"\n最佳模型: {best_model_name}")

# # 保存模型
# joblib.dump(best_model, 'best_obesity_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(le_target, 'label_encoder.pkl')
# print("模型保存完成!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import time
import joblib

# 设置随机种子确保可复现性
np.random.seed(42)

# 1. 加载数据
# 替换为您的实际文件路径
df = pd.read_csv('combined_obesity_data.csv')
print(f"数据集形状: {df.shape}")
print("\n前5行数据:")
print(df.head())
print("\n肥胖类别分布:")
print(df['Obesity_Category'].value_counts())

# 2. 数据预处理 - 修正部分
# 移除可能导致问题的空列
df = df.dropna(axis=1, how='all')

# 检查所有列的数据类型
print("\n原始数据类型:")
print(df.dtypes)

# 识别所有非数值列
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\n非数值列:", non_numeric_cols)

# 编码分类特征 - 包括所有非数值列
categorical_cols = non_numeric_cols.copy()
if 'Obesity_Category' in categorical_cols:
    categorical_cols.remove('Obesity_Category')  # 目标列单独处理

label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        # 确保所有值都是字符串并处理缺失值
        df[col] = df[col].fillna('missing').astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"\n{col}编码映射:")
        print(dict(zip(le.classes_, le.transform(le.classes_))))

# 编码目标变量
le_target = LabelEncoder()
df['Obesity_Category_encoded'] = le_target.fit_transform(df['Obesity_Category'])

# 特征/目标分离
X = df.drop(['Obesity_Category', 'Obesity_Category_encoded'], axis=1)
y = df['Obesity_Category_encoded']

# 确保所有特征都是数值类型
print("\n特征数据类型:")
print(X.dtypes)

# 检查是否还有非数值特征
non_numeric_features = X.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_features:
    print(f"\n警告: 发现非数值特征: {non_numeric_features}")
    print("将对这些特征进行额外编码处理...")
    
    for col in non_numeric_features:
        le = LabelEncoder()
        X[col] = X[col].fillna('missing').astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# 再次检查特征类型
print("\n最终特征数据类型:")
print(X.dtypes)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)

# 特征缩放
scaler = StandardScaler()

# 确保输入是纯数值数组
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# 3. 模型定义和训练
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss'),
    'Naive Bayes': GaussianNB()
}

# 存储结果
results = {
    'Model': [],
    'Accuracy': [],
    'Training Time (s)': [],
    'Prediction Time (s)': [],
    'Cross-Val Score': []
}

# 训练和评估模型
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"训练模型: {name}")
    start_time = time.time()
    
    try:
        # 训练
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # 预测
        start_pred = time.time()
        y_pred = model.predict(X_test_scaled)
        pred_time = time.time() - start_pred
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        
        # 存储结果
        results['Model'].append(name)
        results['Accuracy'].append(accuracy)
        results['Training Time (s)'].append(train_time)
        results['Prediction Time (s)'].append(pred_time)
        results['Cross-Val Score'].append(cv_mean)
        
        # 打印分类报告
        print(f"\n{name} 准确率: {accuracy:.4f}")
        print(f"交叉验证平均准确率: {cv_mean:.4f}")
        print(f"训练时间: {train_time:.4f}秒")
        print(f"预测时间: {pred_time:.6f}秒")
        
        # 打印详细分类报告
        print(f"\n{name} 分类报告:")
        print(classification_report(y_test, y_pred, target_names=le_target.classes_))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le_target.classes_,
                    yticklabels=le_target.classes_)
        plt.title(f'{name} 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"\n训练模型 {name} 时出错: {str(e)}")
        print(f"错误详情: {e}")

# 4. 结果比较
if results['Model']:
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("模型性能比较:")
    print(results_df.sort_values(by='Accuracy', ascending=False))

    # 可视化比较
    plt.figure(figsize=(14, 8))

    # 准确率比较
    plt.subplot(2, 2, 1)
    sns.barplot(x='Accuracy', y='Model', data=results_df.sort_values('Accuracy', ascending=False))
    plt.title('模型准确率比较')
    plt.xlim(0.7, 1.0)

    # 交叉验证准确率比较
    plt.subplot(2, 2, 2)
    sns.barplot(x='Cross-Val Score', y='Model', data=results_df.sort_values('Cross-Val Score', ascending=False))
    plt.title('交叉验证准确率比较')
    plt.xlim(0.7, 1.0)

    # 训练时间比较
    plt.subplot(2, 2, 3)
    sns.barplot(x='Training Time (s)', y='Model', data=results_df.sort_values('Training Time (s)'))
    plt.title('训练时间比较 (秒)')

    # 预测时间比较
    plt.subplot(2, 2, 4)
    sns.barplot(x='Prediction Time (s)', y='Model', data=results_df.sort_values('Prediction Time (s)'))
    plt.title('预测时间比较 (秒)')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()

    # 5. 特征重要性分析 (针对树模型)
    plt.figure(figsize=(12, 8))
    for name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
        if name in models:
            model = models[name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f"{name} - 特征重要性")
                plt.bar(range(X_train.shape[1]), importances[indices], align='center')
                plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png', dpi=300)
                plt.close()

    # 6. 最佳模型保存
    if not results_df.empty:
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        best_model = models[best_model_name]
        print(f"\n最佳模型: {best_model_name}")

        # 保存模型
        joblib.dump(best_model, 'best_obesity_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(le_target, 'label_encoder.pkl')
        print("模型保存完成!")
else:
    print("\n没有成功训练任何模型，请检查数据预处理步骤。")