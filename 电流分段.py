import pandas as pd
from datetime import datetime

# 数据加载
data = pd.read_csv('充电电流.csv')

# 将time列转换为datetime格式
data['time'] = pd.to_datetime(data['time'])

# 定义时间分割点
first_charge_end = datetime.strptime('2024-07-10 11:11', '%Y-%m-%d %H:%M')
second_charge_start = datetime.strptime('2024-07-10 21:10', '%Y-%m-%d %H:%M')
second_charge_end = datetime.strptime('2024-07-11 11:14', '%Y-%m-%d %H:%M')
third_charge_start = datetime.strptime('2024-07-11 21:09', '%Y-%m-%d %H:%M')

# 分割数据
first_charge = data[data['time'] <= first_charge_end]
second_charge = data[(data['time'] >= second_charge_start) & (data['time'] <= second_charge_end)]
third_charge = data[data['time'] >= third_charge_start]

# 输出分段结果
print("First Charge Data:")
print(first_charge)
print("\nSecond Charge Data:")
print(second_charge)
print("\nThird Charge Data:")
print(third_charge)

# 如果需要将分段数据保存到CSV文件中
first_charge.to_csv('一充.csv', index=False)
second_charge.to_csv('二充.csv', index=False)
third_charge.to_csv('三充.csv', index=False)
