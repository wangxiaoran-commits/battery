import pandas as pd


def calculate_constant_voltage_duration(data, voltage_threshold):
    durations = []
    start_time = None
    end_time = None

    for i in range(1, len(data)):
        voltage_diff = abs(data.iloc[i]['val'] - data.iloc[i - 1]['val'])

        if voltage_diff <= voltage_threshold:
            if start_time is None:
                start_time = pd.to_datetime(data.iloc[i - 1]['time'])
            end_time = pd.to_datetime(data.iloc[i]['time'])
        else:
            if start_time is not None and end_time is not None:
                duration = (end_time - start_time).total_seconds()
                durations.append(duration)
            start_time = None
            end_time = None

    if start_time is not None and end_time is not None:
        duration = (end_time - start_time).total_seconds()
        durations.append(duration)

    return durations


# 读取CSV文件
data = pd.read_csv('电压.csv')

# 定义电压阈值
voltage_threshold = 0.01

# 存储所有电池的恒压充电时长
all_durations = {}

# 循环处理每节电池
for i in range(1, 25):
    battery_name = f'单体电池电压2V-{i:03d}电池'
    battery_data = data[data['clique_name'] == battery_name]
    durations = calculate_constant_voltage_duration(battery_data, voltage_threshold)
    all_durations[battery_name] = durations

# 打印结果
for battery, durations in all_durations.items():
    print(f"Battery: {battery}, Constant Voltage Durations (in seconds): {durations}")

#结果
# #durations_list = [
#     252536.0, 254569.0, 253928.0, 254755.0, 254491.0, 252921.0, 253801.0, 253647.0,
#     254245.0, 254270.0, 254017.0, 254336.0, 254342.0, 254296.0, 254569.0, 254563.0,
#     253667.0, 254905.0, 254535.0, 253528.0, 253689.0, 253480.0, 252760.0, 253391.0
# ]