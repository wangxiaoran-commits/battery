import pandas as pd

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

# 识别恒流充电阶段
# 恒流充电阶段通常电流变化不大，电压逐渐上升
current_threshold = 0.05  # 电流变化阈值，根据实际数据调整
voltage_slope_threshold = 0.001  # 电压斜率阈值，根据实际数据调整

ccct_start = None
ccct_end = None

for i in range(1, len(merged_data)):
    current_diff = abs(merged_data['val_current'].iloc[i] - merged_data['val_current'].iloc[i - 1])
    voltage_diff = merged_data['val_voltage'].iloc[i] - merged_data['val_voltage'].iloc[i - 1]

    if current_diff < current_threshold and voltage_diff > voltage_slope_threshold:
        if ccct_start is None:
            ccct_start = merged_data['time'].iloc[i - 1]
        ccct_end = merged_data['time'].iloc[i]
    elif ccct_start is not None:
        break

# 识别恒压充电阶段
# 恒压充电阶段通常电压变化不大，电流逐渐下降
voltage_threshold = 0.01  # 电压变化阈值，根据实际数据调整
current_slope_threshold = -0.001  # 电流斜率阈值，根据实际数据调整

cvct_start = None
cvct_end = None

for i in range(1, len(merged_data)):
    voltage_diff = abs(merged_data['val_voltage'].iloc[i] - merged_data['val_voltage'].iloc[i - 1])
    current_diff = merged_data['val_current'].iloc[i] - merged_data['val_current'].iloc[i - 1]

    if voltage_diff < voltage_threshold and current_diff < current_slope_threshold:
        if cvct_start is None:
            cvct_start = merged_data['time'].iloc[i - 1]
        cvct_end = merged_data['time'].iloc[i]
    elif cvct_start is not None:
        break

# 计算CCCT和CVCT
ccct_duration = (ccct_end - ccct_start).total_seconds() if ccct_start and ccct_end else None
cvct_duration = (cvct_end - cvct_start).total_seconds() if cvct_start and cvct_end else None

print(f"恒流充电时长（CCCT）：{ccct_duration} 秒")
print(f"恒压充电时长（CVCT）：{cvct_duration} 秒")
