import pandas as pd
import numpy as np

# 映射函数
def map_gender(g):
    return 'Male' if g == '男' else 'Female' if g == '女' else g

def map_family_history(h):
    return 'yes' if h == '是' else 'no' if h == '否' else h

# 读取第一个文件 (青少年肥胖问卷)
df1 = pd.read_csv('调查问卷/青少年肥胖与饮食健康调查问卷数据_已清洗.csv')
df1 = df1.rename(columns={
    '性别': 'Gender',
    '年龄': 'Age',
    '身高cm': 'Height',
    '体重kg': 'Weight',
    '是否有家族肥胖史': 'family_history_with_overweight',
    '每周快餐摄入次数': 'FAVC',
    '每日蔬菜水果摄入量份': 'FCVC',
    '每日含糖饮料摄入量瓶': 'CALC',
    '每日饮水量升': 'CH2O',
    '每周运动次数': 'FAF'
})
df1 = df1[['Age', 'Gender', 'Height', 'Weight', 'family_history_with_overweight', 
           'FAVC', 'FCVC', 'CALC', 'CH2O', 'FAF']]

# 单位转换和处理
df1['Height'] = df1['Height'] / 100  # cm转换为米
df1['Gender'] = df1['Gender'].apply(map_gender)
df1['family_history_with_overweight'] = df1['family_history_with_overweight'].apply(map_family_history)
df1['FAVC'] = np.where(df1['FAVC'] > 2, 'yes', 'no')  # 高热量食物
df1['CALC'] = np.where(df1['CALC'] > 1, 'Frequently', 'Sometimes')  # 酒精消费分组

# 读取第二个文件 (删减.csv)
df2 = pd.read_csv('肥胖预测数据收集/处理后.csv')
df2 = df2.rename(columns={
    'Family_history_with_overweight': 'family_history_with_overweight',
    'Favc': 'FAVC',
    'Fcvc': 'FCVC',
    'Ncp': 'NCP',
    'Caec': 'CAEC',
    'Smoke': 'SMOKE',
    'Ch2o': 'CH2O',
    'Scc': 'SCC',
    'Faf': 'FAF',
    'Tue': 'TUE',
    'Calc': 'CALC',
    'Mtrans': 'MTRANS'
})

# 读取第三个文件 (处理后的数据)
df3 = pd.read_csv('肥胖水平/青少年肥胖水平处理过后.csv')

# 统一列名和顺序
common_columns = [
    'Age', 'Gender', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'CALC', 'CH2O', 'CAEC', 'SMOKE', 
    'SCC', 'FAF', 'TUE', 'MTRANS', 'NObeyesdad'
]

# 合并数据集
combined_df = pd.concat([
    df1[df1.columns.intersection(common_columns)],
    df2[df2.columns.intersection(common_columns)],
    df3[df3.columns.intersection(common_columns)]
], ignore_index=True)

# 关键特征处理
combined_df['BMI'] = combined_df['Weight'] / (combined_df['Height'] ** 2)
combined_df['Obesity_Category'] = pd.cut(combined_df['BMI'], 
    bins=[0, 18.5, 25, 30, 35, 40, 100],
    labels=['Underweight', 'Normal', 'Overweight_I', 
            'Overweight_II', 'Obesity_I', 'Obesity_II+'])

# 保存整合后的数据
combined_df.to_csv('combined_obesity_data.csv', index=False)
print("数据集整合完成，保存为 combined_obesity_data.csv")
print(f"总样本数: {len(combined_df)}")