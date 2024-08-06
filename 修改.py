import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import time



def main_function():
    # 数据前期处理
    main_file_path = r"C:\Users\amber\Desktop\特征蓄电池吃\battery_0710-0712.csv"

    # 检查文件是否存在
    if not os.path.exists(main_file_path):
        print(f"File not found: {main_file_path}")
        return
    else:
        def process_chunk(chunk):
            chunk['time'] = pd.to_datetime(chunk['timestamp'], unit='s')
            voltage_data = chunk[chunk['clique_name'].str.contains('单体电池电压2V-')]
            current_data = chunk[chunk['clique_name'].str.contains('充放电电流')]
            return voltage_data, current_data

        voltage_data_total = pd.DataFrame()
        current_data_total = pd.DataFrame()

        chunk_size = 100000
        for chunk in pd.read_csv(main_file_path, chunksize=chunk_size):
            voltage_data_chunk, current_data_chunk = process_chunk(chunk)
            voltage_data_total = pd.concat([voltage_data_total, voltage_data_chunk], ignore_index=True)
            current_data_total = pd.concat([current_data_total, current_data_chunk], ignore_index=True)
        #读取全部的电压与电流数据
        voltage_data_total.to_csv('电压.csv', index=False)
        current_data_total.to_csv('电流.csv', index=False)
    '''
    电压数据处理：
    start_time_str设置想要计算的充电阶段开始时间，duration_hours是本次充电时长
    voltage_file放入已经读取的电压数据'电压.csv'   current_file是'电流.csv'
    peak_time_cutoff限定了快速增长时最大值的选取，是在开始充电后的11小时（利用可视化选取的数据）
    避免峰值取到恒压状态下电流平稳上升到达的最大值，而是取到快速增长情况下的最大值
    filtered_voltage_data取到start_time_str开始持续duration_hours时长的电压数据
    后面处理电流之后，给电流分为First Charge ,Second Charge和Third Charge
    计算恒流充电时长和恒压充电时长用到了’filtered_voltage_data’和‘Third Charge‘数据
    
    并存储peak_time_cutoff，DTW_wasserstein_initial_time_cutoff，DTW_wasserstein_current_time_cutoff'三个时间点
    DTW_wasserstein_initial_time_cutoff设置DTW和wasserstein两次充电循环第一个时间节点，在这个节点之前的数据算作初循环充电曲线
    DTW_wasserstein_current_time_cutoff设置第二个时间节点，在这个节点之后算作当前循环充电曲线
    DTW函数和wasserstein函数可共享这两个电压数据
    '''
    '''
    恒压与恒流的一些阈值：
    0.0025 根据图标可视化选取的阈值，如果每15分钟变化小于0.0025那么从恒流进入恒压阶段
    进入恒压阶段后电压会短期内快速增长到达小高峰，这个max的值作为恒压阶段1和恒压阶段2的分界值
    peak_time_cutoff限定了快速增长时最大值的选取范围，是在开始充电后的11小时（利用可视化选取的数据），避免取到恒压阶段2的最大值
    1.8 ：恒压结束时的电流阈值，当电流小于1.8则恒压结束
    '''

    def segment_current_data(data, segments):
        segmented_data = {}
        for start, end, label in segments:
            if start is None:
                segmented_data[label] = data[data['time'] <= end]
            elif end is None:
                segmented_data[label] = data[data['time'] >= start]
            else:
                segmented_data[label] = data[(data['time'] >= start) & (data['time'] <= end)]
        return segmented_data
    def process_data(voltage_file, current_file, start_time_str, duration_hours):
        data = pd.read_csv(voltage_file)
        data['time'] = pd.to_datetime(data['time'])

        current_data = pd.read_csv(current_file)
        current_data['time'] = pd.to_datetime(current_data['time'])

        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
        end_time = start_time + timedelta(hours=duration_hours)
        peak_time_cutoff = start_time + timedelta(hours=11)

        filtered_voltage_data = data[(data['time'] >= start_time) & (data['time'] <= end_time)].copy()
        filtered_voltage_data.loc[:, 'peak_time_cutoff'] = peak_time_cutoff

        initial_time_cutoff = datetime.strptime('2024-07-10 13:09', '%Y-%m-%d %H:%M')
        current_time_cutoff = datetime.strptime('2024-07-11 13:20', '%Y-%m-%d %H:%M')

        initial_curve = data[(data['time'] >= initial_time_cutoff) & (data['time'] <= datetime.strptime('2024-07-11 11:14', '%Y-%m-%d %H:%M'))]
        current_curve = data[data['time'] >= current_time_cutoff]


        segments = [
            (None, datetime.strptime('2024-07-10 03:11', '%Y-%m-%d %H:%M'), 'First Charge'),
            (datetime.strptime('2024-07-10 13:10', '%Y-%m-%d %H:%M'), datetime.strptime('2024-07-11 03:14', '%Y-%m-%d %H:%M'), 'Second Charge'),
            (datetime.strptime('2024-07-11 13:20', '%Y-%m-%d %H:%M'), None, 'Third Charge')
        ]

        segmented_current_data = segment_current_data(current_data, segments)
        #分次保存电流数据
        for label, data in segmented_current_data.items():
            data.to_csv(f'{label}.csv', index=False)
            print(f"Saved {label} data to {label}.csv")

        first_charge_df = segmented_current_data['First Charge']
        second_charge_df = segmented_current_data['Second Charge']

        return filtered_voltage_data, initial_curve, current_curve, first_charge_df, second_charge_df

    start_time_str = '2024-07-11 13:20'
    duration_hours = 27
    filtered_voltage_data, initial_curve, current_curve, first_charge_df, second_charge_df = process_data('电压.csv',
                                                                                                          '电流.csv',
                                                                                                          start_time_str,
                                                                                                          duration_hours)
    '''
    计算恒压与恒流充电时长
    '''
    def calculate_charge_stages(filtered_voltage_data, current_file):
        data = filtered_voltage_data
        current_data = pd.read_csv(current_file)
        current_data['time'] = pd.to_datetime(current_data['time'])

        stages_dict = {'Battery': [], 'Stage': [], 'Start_Time': [], 'End_Time': []}

        for i in range(1, 25):
            battery_name = f'单体电池电压2V-{i:03d}电池'
            battery_data = data[data['clique_name'] == battery_name].copy()
            peak_time_cutoff = battery_data['peak_time_cutoff'].iloc[0]

            pre_peak_data = battery_data[battery_data['time'] <= peak_time_cutoff]
            if not pre_peak_data.empty:
                max_val = pre_peak_data['val'].max()
                max_idx = pre_peak_data[pre_peak_data['val'] == max_val].index[0]

                steady_rise_start = max_idx
                post_steady_rise_current_data = current_data[current_data['time'] >= battery_data.loc[steady_rise_start]['time']]
                steady_rise_end_time = post_steady_rise_current_data[post_steady_rise_current_data['val'] < 1.8]['time'].min()

                if pd.notna(steady_rise_end_time):
                    steady_rise_end = battery_data[battery_data['time'] >= steady_rise_end_time].index[0]
                else:
                    steady_rise_end = battery_data.index[-1]

                stages_dict['Battery'].append(battery_name)
                stages_dict['Stage'].append('恒压2')
                stages_dict['Start_Time'].append(battery_data.loc[steady_rise_start]['time'])
                stages_dict['End_Time'].append(battery_data.loc[steady_rise_end]['time'])

                first_stage_start = battery_data.index[0]
                first_stage_end = first_stage_start

                while first_stage_end < steady_rise_start:
                    next_point = first_stage_end
                    time_check = battery_data.loc[first_stage_end]['time'] + timedelta(minutes=15)
                    future_points = battery_data[(battery_data['time'] >= battery_data.loc[first_stage_end]['time']) & (battery_data['time'] <= time_check)]
                    if len(future_points) > 1:
                        next_point = future_points.index[-1]
                    if (battery_data.loc[next_point]['time'] - battery_data.loc[first_stage_end]['time']).total_seconds() >= 15 * 60:
                        if abs(battery_data.loc[next_point]['val'] - battery_data.loc[first_stage_end]['val']) <= 0.0025:
                            break
                    first_stage_end = next_point

                stages_dict['Battery'].append(battery_name)
                stages_dict['Stage'].append('恒流')
                stages_dict['Start_Time'].append(battery_data.loc[first_stage_start]['time'])
                stages_dict['End_Time'].append(battery_data.loc[first_stage_end]['time'])

                slow_growth_start = first_stage_end + 1
                slow_growth_end = steady_rise_start - 1

                if slow_growth_start <= slow_growth_end:
                    stages_dict['Battery'].append(battery_name)
                    stages_dict['Stage'].append('恒压1')
                    stages_dict['Start_Time'].append(battery_data.loc[slow_growth_start]['time'])
                    stages_dict['End_Time'].append(battery_data.loc[slow_growth_end]['time'])
                else:
                    stages_dict['Battery'].append(battery_name)
                    stages_dict['Stage'].append('恒流')
                    stages_dict['Start_Time'].append(None)
                    stages_dict['End_Time'].append(None)
            else:
                stages_dict['Battery'].append(battery_name)
                stages_dict['Stage'].append('恒压2')
                stages_dict['Start_Time'].append(None)
                stages_dict['End_Time'].append(None)

        stages_df = pd.DataFrame(stages_dict)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 1000)

        return stages_df

    stages_df = calculate_charge_stages(filtered_voltage_data, 'Third Charge.csv')
    print(stages_df)
    """
    计算每节电池在恒压充电阶段的电压变化率平稳时长（CVCT）

    参数:
    data (DataFrame): 包含电压数据的DataFrame，必须包含以下列:
        - 'timestamp': 时间戳
        - 'clique_name': 电池名称
        - 'val': 电压值

    返回:
    DataFrame: 包含每节电池CVCT值的DataFrame，列为:
        - 'Battery': 电池名称
        - 'CVCT': 电压变化率平稳时长（秒）

    详细说明:
    1. 遍历每节电池的数据，提取其时间戳和电压值。
    2. 计算时间差和电压差，以获得电压变化率（dV/dt）。
    3. 确定电压变化率在平稳区域（-0.0001 <= dV/dt <= 0.0002）内的时间段。
    4. 计算平稳区域的持续时间，并找出最长的平稳时间段。
    5. 返回一个包含每节电池名称及其CVCT值的DataFrame。
    """
    def calculate_cvct(filtered_voltage_data):
        start_time = time.time()
        batteries = [f'单体电池电压2V-{i:03d}电池' for i in range(1, 25)]
        cvct_dict = {}

        for battery in batteries:
            battery_data = filtered_voltage_data[filtered_voltage_data['clique_name'] == battery].reset_index(drop=True)

            if len(battery_data) > 1:
                # 计算时间差和电压差
                time_diff = np.diff(battery_data['timestamp'])
                voltage_diff = np.diff(battery_data['val'])
                dv_dt = voltage_diff / time_diff
                 # 找到平稳区域（-0.0001 <= dV/dt <= 0.0002）
                stable_region = (dv_dt >= -0.0001) & (dv_dt <= 0.0002)
                # 计算平稳区域的持续时间
                stable_durations = []
                current_duration = 0

                for i in range(len(stable_region)):
                    if stable_region[i]:
                        current_duration += time_diff[i]
                    else:
                        if current_duration > 0:
                            stable_durations.append(current_duration)
                            current_duration = 0
                # 添加最后一个持续时间
                if current_duration > 0:
                    stable_durations.append(current_duration)
                # 选取最长的平稳区域持续时间
                if stable_durations:
                    cvct = max(stable_durations)
                else:
                    cvct = np.nan # 如果没有平稳区域，返回NaN
            else:
                cvct = np.nan   # 如果数据点不足，返回NaN

            cvct_dict[battery] = cvct
        end_time = time.time()
        print(f"calculate_cvct function execution time: {end_time - start_time} seconds")
        return pd.DataFrame(list(cvct_dict.items()), columns=['Battery', 'CVCT'])

    def calculate_max_dv_dt(filtered_voltage_data):
        start_time = time.time()
        """
           计算每节电池在恒压充电阶段的最大电压变化率（dV/dt）。

           参数:
           data (DataFrame): 包含电压数据的DataFrame，必须包含以下列:
               - 'timestamp': 时间戳
               - 'clique_name': 电池名称
               - 'val': 电压值

           返回:
           DataFrame: 包含每节电池最大dV/dt值的DataFrame，列为:
               - 'Battery': 电池名称
               - 'Max_dV/dt': 最大电压变化率（dV/dt）

           详细说明:
           1. 遍历每节电池的数据，提取其时间戳和电压值。
           2. 计算时间差和电压差，以获得电压变化率（dV/dt）。
           3. 找到电压变化率的最大值。
           4. 返回一个包含每节电池名称及其最大dV/dt值的DataFrame。
           """
        batteries = [f'单体电池电压2V-{i:03d}电池' for i in range(1, 25)]
        max_dv_dt_dict = {}

        for battery in batteries:
            battery_data = filtered_voltage_data[filtered_voltage_data['clique_name'] == battery]

            if len(battery_data) > 1:
                time_diff = np.diff(battery_data['timestamp'])
                voltage_diff = np.diff(battery_data['val'])
                dv_dt = voltage_diff / time_diff
                # 找到最大dV/dt值
                max_dv_dt = np.max(dv_dt)
            else:
                max_dv_dt = np.nan

            max_dv_dt_dict[battery] = max_dv_dt
        end_time = time.time()
        print(f"calculate_max_dv_dt function execution time: {end_time - start_time} seconds")

        return pd.DataFrame(list(max_dv_dt_dict.items()), columns=['Battery', 'Max_dV/dt'])

    def calculate_dtw(initial_curve, current_curve):
        start_time = time.time()

        def dtw(A, B):
            n, m = len(A), len(B)
            M = np.zeros((n, m))

            for i in range(n):
                for j in range(m):
                    M[i, j] = (A[i] - B[j]) ** 2

            r = np.zeros((n, m))
            r[0, 0] = M[0, 0]

            for i in range(1, n):
                r[i, 0] = M[i, 0] + r[i - 1, 0]
            for j in range(1, m):
                r[0, j] = M[0, j] + r[0, j - 1]
            for i in range(1, n):
                for j in range(1, m):
                    r[i, j] = M[i, j] + min(r[i - 1, j - 1], r[i, j - 1], r[i - 1, j])

            V_DTW = r[n - 1, m - 1]
            return V_DTW

        dtw_distances = []
        for i in range(1, 25):
            battery_name = f'单体电池电压2V-{i:03d}电池'
            initial_curve_battery = initial_curve[initial_curve['clique_name'] == battery_name]['val'].values
            current_curve_battery = current_curve[current_curve['clique_name'] == battery_name]['val'].values

            if len(initial_curve_battery) > 0 and len(current_curve_battery) > 0:
                dtw_distance = dtw(initial_curve_battery, current_curve_battery)
                dtw_distances.append((battery_name, dtw_distance))
            else:
                dtw_distances.append((battery_name, None))

        end_time = time.time()
        print(f"calculate_dtw function execution time: {end_time - start_time} seconds")

        return pd.DataFrame(dtw_distances, columns=['Battery', 'DTW_distance'])





    def calculate_wasserstein_distance(initial_curve, current_curve):
        start_time = time.time()

        wasserstein_distances = []
        for i in range(1, 25):
            battery_name = f'单体电池电压2V-{i:03d}电池'
            initial_curve_battery = initial_curve[initial_curve['clique_name'] == battery_name]['val'].values
            current_curve_battery = current_curve[current_curve['clique_name'] == battery_name]['val'].values

            if len(initial_curve_battery) > 0 and len(current_curve_battery) > 0:
                distance = wasserstein_distance(initial_curve_battery, current_curve_battery)
                wasserstein_distances.append((battery_name, distance))
            else:
                wasserstein_distances.append((battery_name, None))

        end_time = time.time()
        print(f"calculate_wasserstein_distance function execution time: {end_time - start_time} seconds")

        return pd.DataFrame(wasserstein_distances, columns=['Battery', 'Wasserstein_distance'])

    def inner_function_vis():
        voltage_file = '电压.csv'
        current_file = '电流.csv'

        voltage_data = pd.read_csv(voltage_file)
        current_data = pd.read_csv(current_file)

        voltage_data['time'] = pd.to_datetime(voltage_data['time'])
        current_data['time'] = pd.to_datetime(current_data['time'])

        fig, ax1 = plt.subplots(figsize=(15, 10))

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Current (A)', color='tab:blue')
        ax1.plot(current_data['time'], current_data['val'], color='tab:blue', label='Current')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(which='both', linestyle='--', linewidth=0.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Voltage (V)', color='tab:red')

        for column_name in voltage_data['clique_name'].unique():
            if column_name.startswith('单体电池电压2V-') and column_name.endswith('电池'):
                battery_data = voltage_data[voltage_data['clique_name'] == column_name]
                ax2.plot(battery_data['time'], battery_data['val'], label=column_name)

        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.02))
        ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        plt.title('Current and Voltage over Time')
        plt.show()

    inner_function_vis()

    cvct_df = calculate_cvct(filtered_voltage_data)

    max_dv_dt_df = calculate_max_dv_dt(filtered_voltage_data)

    dtw_distances_df = calculate_dtw(initial_curve, current_curve)

    wasserstein_distances_df = calculate_wasserstein_distance(initial_curve, current_curve)
    combined_df = pd.merge(cvct_df, max_dv_dt_df, on='Battery')
    combined_df = pd.merge(combined_df, dtw_distances_df, on='Battery')
    combined_df = pd.merge(combined_df, wasserstein_distances_df, on='Battery')

    print(combined_df)

main_function()
