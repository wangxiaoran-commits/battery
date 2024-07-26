import pandas as pd
import numpy as np
from datetime import datetime

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


# 定义DTW函数
def dtw(A, B):
    n, m = len(A), len(B)
    M = np.zeros((n, m))

    # 计算代价矩阵
    for i in range(n):
        for j in range(m):
            M[i, j] = (A[i] - B[j]) ** 2

    # 初始化累积距离矩阵
    r = np.zeros((n, m))
    r[0, 0] = M[0, 0]

    # 使用动态规划填充累积距离矩阵
    for i in range(1, n):
        r[i, 0] = M[i, 0] + r[i - 1, 0]
    for j in range(1, m):
        r[0, j] = M[0, j] + r[0, j - 1]
    for i in range(1, n):
        for j in range(1, m):
            r[i, j] = M[i, j] + min(r[i - 1, j - 1], r[i, j - 1], r[i - 1, j])

    # DTW 距离
    V_DTW = r[n - 1, m - 1]
    return V_DTW



dtw_distances = []

# 计算每个电池的DTW距离
for i in range(1, 11):
    battery_name = f'单体电池电压2V-{i:03d}电池'

    initial_curve_battery = initial_curve[initial_curve['clique_name'] == battery_name]['val'].values
    current_curve_battery = current_curve[current_curve['clique_name'] == battery_name]['val'].values

    if len(initial_curve_battery) > 0 and len(current_curve_battery) > 0:
        dtw_distance = dtw(initial_curve_battery, current_curve_battery)
        dtw_distances.append((battery_name, dtw_distance))
    else:
        dtw_distances.append((battery_name, None))

# 转换为DataFrame
result_df = pd.DataFrame(dtw_distances, columns=['clique_name', 'DTW_distance'])
print(result_df)
