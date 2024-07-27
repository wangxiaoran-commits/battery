# import pandas as pd
#
# # 读取电压和电流数据
# voltage_file = '电压.csv'
# current_file = '充电电流.csv'
#
# voltage_data = pd.read_csv(voltage_file)
# current_data = pd.read_csv(current_file)
#
# # 确保时间格式正确
# voltage_data['time'] = pd.to_datetime(voltage_data['timestamp'], unit='s')
# current_data['time'] = pd.to_datetime(current_data['timestamp'], unit='s')
#
# # 按时间合并电压和电流数据
# merged_data = pd.merge_asof(voltage_data.sort_values('time'),
#                             current_data.sort_values('time'),
#                             on='time',
#                             suffixes=('_voltage', '_current'))
#
# # 识别恒流充电阶段
# # 恒流充电阶段通常电流变化不大，电压逐渐上升
# current_threshold = 3  # 电流变化阈值，根据实际数据调整
# voltage_slope_threshold = 0.09# 电压斜率阈值，根据实际数据调整
#
# ccct_start = None
# ccct_end = None
#
# for i in range(1, len(merged_data)):
#     current_diff = abs(merged_data['val_current'].iloc[i] - merged_data['val_current'].iloc[i - 1])
#     voltage_diff = merged_data['val_voltage'].iloc[i] - merged_data['val_voltage'].iloc[i - 1]
#
#     if current_diff < current_threshold and voltage_diff > voltage_slope_threshold:
#         if ccct_start is None:
#             ccct_start = merged_data['time'].iloc[i - 1]
#         ccct_end = merged_data['time'].iloc[i]
#     elif ccct_start is not None:
#         break
#
# # 识别恒压充电阶段
# # 恒压充电阶段通常电压变化不大，电流逐渐下降
# voltage_threshold = 3 # 电压变化阈值，根据实际数据调整
# current_slope_threshold = -0.008 # 电流斜率阈值，根据实际数据调整
#
# cvct_start = None
# cvct_end = None
#
# for i in range(1, len(merged_data)):
#     voltage_diff = abs(merged_data['val_voltage'].iloc[i] - merged_data['val_voltage'].iloc[i - 1])
#     current_diff = merged_data['val_current'].iloc[i] - merged_data['val_current'].iloc[i - 1]
#
#     if voltage_diff < voltage_threshold and current_diff < current_slope_threshold:
#         if cvct_start is None:
#             cvct_start = merged_data['time'].iloc[i - 1]
#         cvct_end = merged_data['time'].iloc[i]
#     elif cvct_start is not None:
#         break
#
# # 计算CCCT和CVCT
# ccct_duration = (ccct_end - ccct_start).total_seconds() if ccct_start and ccct_end else None
# cvct_duration = (cvct_end - cvct_start).total_seconds() if cvct_start and cvct_end else None
#
# print(f"恒流充电时长（CCCT）：{ccct_duration} 秒")
# print(f"恒压充电时长（CVCT）：{cvct_duration} 秒")
import pandas as pd

# 计算充电时长的函数
def calculate_constant_current_duration(data, current_threshold=0.1):
    """
    计算恒流充电时长的函数。

    参数:
    data (DataFrame): 包含电流数据的DataFrame，必须包含 'time' 和 'val' 列。
    current_threshold (float): 电流变化的阈值，小于该阈值的变化视为恒流。

    返回:
    list: 包含恒流充电时长的列表（秒）。
    """
    start_time = None
    end_time = None
    durations = []

    for i in range(1, len(data)):
        current_diff = abs(data.iloc[i]['val'] - data.iloc[i - 1]['val'])
        if current_diff <= current_threshold:
            if start_time is None:
                start_time = pd.to_datetime(data.iloc[i - 1]['time'])
            end_time = pd.to_datetime(data.iloc[i]['time'])
        else:
            if start_time is not None and end_time is not None:
                duration = (end_time - start_time).total_seconds()
                durations.append(duration)
            start_time = None
            end_time = None

    # 如果最后一段也是恒流的，需要记录下来
    if start_time is not None and end_time is not None:
        duration = (end_time - start_time).total_seconds()
        durations.append(duration)

    return durations

# 处理文件的函数
def process_file(file_name, current_threshold=0.1):
    """
    处理指定文件并计算恒流充电时长。

    参数:
    file_name (str): 文件名。
    current_threshold (float): 电流变化的阈值，小于该阈值的变化视为恒流。

    返回:
    tuple: 恒流充电时长的列表（秒）和总的恒流充电时间（秒）。
    """
    # 加载数据
    data = pd.read_csv(file_name)

    # 确保数据按照时间排序
    data = data.sort_values(by='time')

    # 计算恒流充电时长
    constant_current_durations = calculate_constant_current_duration(data, current_threshold)

    total_constant_current_duration = sum(constant_current_durations)

    return constant_current_durations, total_constant_current_duration

# 文件列表
file_list = ['二充.csv', '三充.csv']

# 处理每个文件
for file_name in file_list:
    durations, total_duration = process_file(file_name)

    # 输出结果
    print(f"\n文件: {file_name}")
    for idx, duration in enumerate(durations):
        print(f"恒流充电时长 {idx + 1}: {duration} 秒")
    print(f"总的恒流充电时间: {total_duration} 秒")
