import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
print("当前工作目录:", os.getcwd())
file_path = '数据整合.csv'
print("文件路径:", os.path.abspath(file_path))
print("文件存在:", os.path.exists(file_path))

# 2. 更可靠地读取数据 - 处理可能的编码问题
try:
    # 尝试多种可能的编码格式
    for encoding in ['utf-8', 'gbk', 'latin1', 'ISO-8859-1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"使用 {encoding} 编码成功读取数据")
            break
        except:
            continue
    else:
        # 如果所有编码都失败，使用errors='replace'
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
        print("使用错误替换模式读取数据")
except Exception as e:
    print(f"读取文件时发生错误: {e}")
    exit()

# 3. 打印数据的前几行和列名
print("\n数据预览:")
print(df.head(3))
print("\n所有列名:", df.columns.tolist())

# 遍历所有列，处理数值型数据
for col in df.columns:
    # 检查列是否为浮点类型
    if df[col].dtype == np.float64:
        # 保留两位小数
        df[col] = df[col].round(2)
    
    # 处理数值类型但不是浮点型的情况（如整数转换为浮点）
    elif np.issubdtype(df[col].dtype, np.integer):
        # 先转换为浮点型再保留两位小数
        df[col] = df[col].astype(float).round(2)
    
    # 处理可能包含数值的字符串（如果字符串可以转为数值）
    elif df[col].dtype == object:
        try:
            # 尝试转换为浮点数
            converted = pd.to_numeric(df[col], errors='coerce')
            # 保留两位小数，并替换原列中可转换的值
            df[col] = converted.round(2).fillna(df[col])
        except:
            # 如果转换失败，保持原样
            pass


# 根据表头解释创建列名映射（中英文对照）
column_mapping = {
    '性别': 'Gender',
    '年龄': 'Age',
    '高度': 'Height',
    '重量': 'Weight',
    'family_history': 'Family_history',
    'FAVC': 'FAVC',
    'FCVC': 'FCVC',
    'NCP': 'NCP',
    'CAEC': 'CAEC',
    'SMOKE': 'SMOKE',
    'CH2O': 'CH2O',
    'SCC': 'SCC',
    'FAF': 'FAF',
    'TUE': 'TUE',
    'CALC': 'CALC',
    'MTRANS': 'MTRANS',
    'Obesity_level': 'Obesity_level'
}
# 统一列名为英文
df = df.rename(columns={cn: column_mapping[cn] for cn in df.columns if cn in column_mapping})

# 1. 计算BMI (体重/身高^2)
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# 2. 创建年龄分组
bins = [0, 18, 30, 40, 100]
labels = ['Teen', 'Young_Adult', 'Adult', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# 3. 创建BMI分类 (根据WHO标准)
def bmi_classification(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    elif 30 <= bmi < 35:
        return 'Obesity_I'
    elif 35 <= bmi < 40:
        return 'Obesity_II'
    else:
        return 'Obesity_III'

df['BMI_Class'] = df['BMI'].apply(bmi_classification)

# 4. 二值特征编码（映射为0/1）
binary_cols = ['Family_history', 'FAVC', 'SMOKE', 'SCC']
binary_mapping = {'yes': 1, 'no': 0, '是': 1, '否': 0}
for col in binary_cols:
    df[col] = df[col].map(binary_mapping).fillna(df[col])

# 5. 有序分类特征编码（使用标签编码）
ordinal_cols = ['CAEC', 'CALC']
ordinal_mapping = {
    'no': 0, '否': 0,
    'Sometimes': 1, '偶尔': 1,
    'Frequently': 2, '经常': 2,
    'Always': 3, '总是': 3
}
for col in ordinal_cols:
    df[col] = df[col].map(ordinal_mapping).fillna(df[col])

# 6. 高热量食物和蔬菜摄入的交互特征
df['HighCal_Veg_Interaction'] = df['FAVC'] * df['FCVC']

# 7. 体力活动与技术使用的比率
df['PA_Tech_Ratio'] = df['FAF'] / (df['TUE'] + 0.001)  # 避免除零

# 8. 分类型交通方式的编码
trans_mapping = {
    'Automobile': 0, '汽车': 0,
    'Motorbike': 1, '摩托车': 1,
    'Bike': 2, '自行车': 2,
    'Public_Transportation': 3, '公共交通': 3,
    'Walking': 4, '步行': 4
}
df['MTRANS'] = df['MTRANS'].map(trans_mapping).fillna(df['MTRANS'])

# 9. 年龄与BMI的交互
df['Age_BMI_Interaction'] = df['Age'] * df['BMI']

# 10. 饮食特征组合
df['Meal_Density'] = df['NCP'] / (df['FCVC'] + 0.001)  # 每餐平均蔬菜摄入

# 11. 对BMI分类和年龄组进行独热编码
df = pd.get_dummies(df, columns=['BMI_Class', 'Age_Group'], prefix=['BMI', 'AgeGrp'])

# 打印新增特征
print("\n新增特征列表:")
new_features = [col for col in df.columns if col not in column_mapping.values()]
print(new_features)

# 保存处理后的数据
df.to_csv('数据整合_特征工程.csv', index=False)
print("特征工程完成，结果已保存到'数据整合_特征工程.csv'")