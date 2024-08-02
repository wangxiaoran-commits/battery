import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import numpy as np
np.set_printoptions(threshold=np.inf)
# 读取所有电压数据
data = pd.read_csv('电压.csv')

# 将time列转换为datetime格式
data['time'] = pd.to_datetime(data['time'])

# 定义时间分割点
current_time_cutoff = datetime.strptime('2024-07-11 13:20', '%Y-%m-%d %H:%M')
peak_time_cutoff = datetime.strptime('2024-07-12 00:00', '%Y-%m-%d %H:%M')

# 过滤出时间大于等于current_time_cutoff的数据
filtered_data = data[data['time'] >= current_time_cutoff]

# 初始化存储结果的字典
stages_dict = {'Battery': [], 'Stage': [], 'Start_Time': [], 'End_Time': []}

# 分析每节电池
for i in range(1, 25):
    battery_name = f'单体电池电压2V-{i:03d}电池'
    battery_data = filtered_data[filtered_data['clique_name'] == battery_name].copy()

    # 找到在peak_time_cutoff之前的电压最大值
    pre_peak_data = battery_data[battery_data['time'] <= peak_time_cutoff]
    if not pre_peak_data.empty:
        max_val = pre_peak_data['val'].max()
        max_idx = pre_peak_data[pre_peak_data['val'] == max_val].index[0]

        # 第三阶段的起始和结束点
        steady_rise_start = max_idx
        steady_rise_end = battery_data.index[-1]

        # 存储第三阶段信息
        stages_dict['Battery'].append(battery_name)
        stages_dict['Stage'].append('steady_rise')
        stages_dict['Start_Time'].append(battery_data.loc[steady_rise_start]['time'])
        stages_dict['End_Time'].append(battery_data.loc[steady_rise_end]['time'])

        # 计算第一阶段
        first_stage_start = battery_data.index[0]
        first_stage_end = first_stage_start

        while first_stage_end < steady_rise_start:
            next_point = first_stage_end
            # 找到下一个十五分钟的点
            time_check = battery_data.loc[first_stage_end]['time'] + timedelta(minutes=15)
            future_points = battery_data[(battery_data['time'] >= battery_data.loc[first_stage_end]['time']) & (
                        battery_data['time'] <= time_check)]
            if len(future_points) > 1:
                next_point = future_points.index[-1]
            if (battery_data.loc[next_point]['time'] - battery_data.loc[first_stage_end][
                'time']).total_seconds() >= 15 * 60:
                if abs(battery_data.loc[next_point]['val'] - battery_data.loc[first_stage_end]['val']) <= 0.004:
                    break
            first_stage_end = next_point

        # 存储第一阶段信息
        stages_dict['Battery'].append(battery_name)
        stages_dict['Stage'].append('fast_growth')
        stages_dict['Start_Time'].append(battery_data.loc[first_stage_start]['time'])
        stages_dict['End_Time'].append(battery_data.loc[first_stage_end]['time'])

        # 第二阶段的起始和结束点
        slow_growth_start = first_stage_end + 1
        slow_growth_end = steady_rise_start - 1

        # 存储第二阶段信息
        if slow_growth_start <= slow_growth_end:
            stages_dict['Battery'].append(battery_name)
            stages_dict['Stage'].append('slow_growth')
            stages_dict['Start_Time'].append(battery_data.loc[slow_growth_start]['time'])
            stages_dict['End_Time'].append(battery_data.loc[slow_growth_end]['time'])
    else:
        # 如果没有找到最大值，则记录该电池没有第三阶段
        stages_dict['Battery'].append(battery_name)
        stages_dict['Stage'].append('steady_rise')
        stages_dict['Start_Time'].append(None)
        stages_dict['End_Time'].append(None)

    # 转换为DataFrame
stages_df = pd.DataFrame(stages_dict)

    # 打印阶段划分结果
print(stages_df)