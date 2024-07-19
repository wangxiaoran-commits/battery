import pandas as pd
import numpy as np

# 读取电压和电流数据
voltage_file = '电压.csv'
current_file = '充电电流.csv'

voltage_data = pd.read_csv(voltage_file)
current_data = pd.read_csv(current_file)

# 确保时间格式正确
voltage_data['time'] = pd.to_datetime(voltage_data['timestamp'], unit='s')
current_data['time'] = pd.to_datetime(current_data['timestamp'], unit='s')

# 按时间合并电压和电流数据
merged_data = pd.merge_asof(voltage_data.sort_values('time'),
                            current_data.sort_values('time'),
                            on='time',
                            suffixes=('_voltage', '_current'))

# 计算能量变化 E(t) = ∫ V(τ) * I(τ) dτ
# 使用梯形法进行数值积分
merged_data['energy'] = (merged_data['val_voltage'] * merged_data['val_current']).cumsum() * (merged_data['time'].diff().dt.total_seconds().fillna(0))

# 计算最大可用容量 C_now
C_now = merged_data['energy'].max()

print(f"当前电池的最大可用容量 C_now: {C_now} Wh")

# 保存计算结果
merged_data.to_csv('merged_data_with_energy.csv', index=False)
