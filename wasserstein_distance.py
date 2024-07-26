import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import wasserstein_distance

# 读取所有电压数据
data = pd.read_csv('电压.csv')

# 将time列转换为datetime格式
data['time'] = pd.to_datetime(data['time'])

# 定义时间分割点
initial_time_cutoff = datetime.strptime('2024-07-11 11:15', '%Y-%m-%d %H:%M')
current_time_cutoff = datetime.strptime('2024-07-11 21:20', '%Y-%m-%d %H:%M')

# 分割数据
initial_curve = data[data['time'] <= initial_time_cutoff]
current_curve = data[data['time'] >= current_time_cutoff]

# 初始化存储结果的列表
wasserstein_distances = []

# 计算每个电池的Wasserstein距离
for i in range(1, 25):
    battery_name = f'单体电池电压2V-{i:03d}电池'

    initial_curve_battery = initial_curve[initial_curve['clique_name'] == battery_name]['val'].values
    current_curve_battery = current_curve[current_curve['clique_name'] == battery_name]['val'].values

    if len(initial_curve_battery) > 0 and len(current_curve_battery) > 0:
        distance = wasserstein_distance(initial_curve_battery, current_curve_battery)
        wasserstein_distances.append((battery_name, distance))
    else:
        wasserstein_distances.append((battery_name, None))

# 转换为DataFrame
result_df = pd.DataFrame(wasserstein_distances, columns=['clique_name', 'Wasserstein_distance'])
print(result_df)

