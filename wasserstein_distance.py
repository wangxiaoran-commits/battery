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

#        clique_name  Wasserstein_distance
# 0   单体电池电压2V-001电池              0.095847
# 1   单体电池电压2V-002电池              0.092427
# 2   单体电池电压2V-003电池              0.089752
# 3   单体电池电压2V-004电池              0.089336
# 4   单体电池电压2V-005电池              0.090359
# 5   单体电池电压2V-006电池              0.092766
# 6   单体电池电压2V-007电池              0.093395
# 7   单体电池电压2V-008电池              0.089415
# 8   单体电池电压2V-009电池              0.090230
# 9   单体电池电压2V-010电池              0.089835
# 10  单体电池电压2V-011电池              0.090720
# 11  单体电池电压2V-012电池              0.096739
# 12  单体电池电压2V-013电池              0.088564
# 13  单体电池电压2V-014电池              0.089338
# 14  单体电池电压2V-015电池              0.090804
# 15  单体电池电压2V-016电池              0.089715
# 16  单体电池电压2V-017电池              0.089926
# 17  单体电池电压2V-018电池              0.092487
# 18  单体电池电压2V-019电池              0.090957
# 19  单体电池电压2V-020电池              0.091230
# 20  单体电池电压2V-021电池              0.091836
# 21  单体电池电压2V-022电池              0.091517
# 22  单体电池电压2V-023电池              0.091876
# 23  单体电池电压2V-024电池              0.092159