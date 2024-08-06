import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from datetime import datetime, timedelta

def main_function():
    # 数据前期处理
    # 文件路径
    main_file_path = r"C:\Users\amber\Desktop\特征蓄电池吃\battery_0710-0712.csv"

    # 检查文件是否存在
    if not os.path.exists(main_file_path):
        print(f"File not found: {main_file_path}")
        return
    else:
        # 定义处理每个chunk的函数
        def process_chunk(chunk):
            # 将 'timestamp' 从 Unix 时间转换为 datetime
            chunk['time'] = pd.to_datetime(chunk['timestamp'], unit='s')

            # 过滤并分类数据
            voltage_data = chunk[chunk['clique_name'].str.contains('单体电池电压2V-')]
            internal_resistance_data = chunk[chunk['clique_name'].str.contains('单体电池电压2V内阻-')]
            current_data = chunk[chunk['clique_name'].str.contains('充放电电流')]

            return voltage_data, internal_resistance_data, current_data

        # 新建DataFrame
        voltage_data_total = pd.DataFrame()
        internal_resistance_data_total = pd.DataFrame()
        current_data_total = pd.DataFrame()

        # 分块读取和处理
        chunk_size = 100000

        # 分块读取
        for chunk in pd.read_csv(main_file_path, chunksize=chunk_size):
            voltage_data_chunk, internal_resistance_data_chunk, current_data_chunk = process_chunk(chunk)

            # 处理后的chunk加到总数据
            voltage_data_total = pd.concat([voltage_data_total, voltage_data_chunk], ignore_index=True)
            internal_resistance_data_total = pd.concat([internal_resistance_data_total, internal_resistance_data_chunk], ignore_index=True)
            current_data_total = pd.concat([current_data_total, current_data_chunk], ignore_index=True)

        # 将原始数据根据变量名分为‘电压’，‘电流’和‘电阻’并储存下来
        voltage_data_total.to_csv('电压.csv', index=False)
        internal_resistance_data_total.to_csv('电阻.csv', index=False)
        current_data_total.to_csv('电流.csv', index=False)


    '''自己设定分割时间点，分割想要利用的充电电压数据，分割前的数据被存储在'电压.csv'中，
    分割后的充电电压数据被存储在'filtered_voltage_data.csv'文件里面
    之后利用'filtered_voltage_data.csv'计算想要的特征，比如恒流、恒压充电时长
    start_time_str代表想要开始 duration_hours代表本次充电需要计算的持续时间'''

    def extract_voltage_data(voltage_file, current_file, start_time_str, duration_hours):
        # 读取所有电压数据
        data = pd.read_csv(voltage_file)
        data['time'] = pd.to_datetime(data['time'])

        # 读取电流数据
        current_data = pd.read_csv(current_file)
        current_data['time'] = pd.to_datetime(current_data['time'])

        # 定义时间区间
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
        end_time = start_time + timedelta(hours=duration_hours)

        # 设置peak_time_cutoff为放电开始后的十一个小时
        peak_time_cutoff = start_time + timedelta(hours=11)

        # 定义DTW和Wasserstein时间分割点
        DTW_wasserstein_initial_time_cutoff = datetime.strptime('2024-07-11 03:15', '%Y-%m-%d %H:%M')
        DTW_wasserstein_current_time_cutoff = datetime.strptime('2024-07-11 13:20', '%Y-%m-%d %H:%M')

        # 过滤出时间区间内的数据
        filtered_voltage_data = data[(data['time'] >= start_time) & (data['time'] <= end_time)]

        # 将时间分割点添加到数据中
        filtered_voltage_data['peak_time_cutoff'] = peak_time_cutoff
        filtered_voltage_data['DTW_wasserstein_initial_time_cutoff'] = DTW_wasserstein_initial_time_cutoff
        filtered_voltage_data['DTW_wasserstein_current_time_cutoff'] = DTW_wasserstein_current_time_cutoff

        # 将过滤后的数据保存为新的CSV文件
        filtered_voltage_data.to_csv('filtered_voltage_data.csv', index=False)

        print("Filtered voltage data saved to 'filtered_voltage_data.csv'")

        return filtered_voltage_data

    # 定义时间区间
    start_time_str = '2024-07-11 13:20'
    duration_hours = 27 # 根据需要修改时间区间
    # 提取时间区间内的电压数据
    filtered_voltage_data = extract_voltage_data('电压.csv', '电流.csv', start_time_str, duration_hours)




'''根据时间点给电流数据分段，原来的电流数据被存储在‘电流.csv’文件里，分割完会被分为
    ‘First Charge’,'Second Charge','Third Charge' (这里设置了三段，可以根据实际情况修改)
    在计算恒流和恒压部分用到Third Charge.csv'''

      # 电流数据的分段处理
    def segment_current_data(data, segments):
        """
        根据指定的时间段对数据进行分隔，以便分段计算恒流充电时间，作为几个特征值

        参数:
        data (DataFrame): 包含时间和电流数据的DataFrame
        segments (list of tuples): 每个元组包含一个时间段的开始和结束时间

        返回:
        dict: 每个键是时间段，值是对应时间段内的数据
        """
        segmented_data = {}
        for start, end, label in segments:
            if start is None:
                segmented_data[label] = data[data['time'] <= end]
            elif end is None:
                segmented_data[label] = data[data['time'] >= start]
            else:
                segmented_data[label] = data[(data['time'] >= start) & (data['time'] <= end)]
        return segmented_data

    # 定义时间分割点（手动设置）
    segments = [
        (None, datetime.strptime('2024-07-10 03:11', '%Y-%m-%d %H:%M'), 'First Charge'),
        (datetime.strptime('2024-07-10 13:10', '%Y-%m-%d %H:%M'), datetime.strptime('2024-07-11 03:14', '%Y-%m-%d %H:%M'), 'Second Charge'),
        (datetime.strptime('2024-07-11 13:20', '%Y-%m-%d %H:%M'), None, 'Third Charge')
    ]

    # 原始电流数据，要求有‘time’，‘clique_name’,'val','timestamp'列
    current_data = pd.read_csv('电流.csv')

    # 确保数据按时间排序
    current_data['time'] = pd.to_datetime(current_data['time'])
    current_data = current_data.sort_values(by='time')

    # 分段数据
    segmented_current_data = segment_current_data(current_data, segments)

    # 保存分段数据，在计算恒流充电时间时直接使用
    for label, data in segmented_current_data.items():
        data.to_csv(f'{label}.csv', index=False)
        print(f"Saved {label} data to {label}.csv")


'''恒流、恒压充电时长计算（这里主动选取第二次充电进行计算，若要计算其他充电时间的数据，需要手动修改时间分割点）
        filtered_voltage_data和current_file是一次充电的电压数据
        在之前的数据处理过程中自己设置时间节点进行提取，current_file取得是Third Charge.csv
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
                post_steady_rise_current_data = current_data[
                    current_data['time'] >= battery_data.loc[steady_rise_start]['time']]
                steady_rise_end_time = post_steady_rise_current_data[post_steady_rise_current_data['val'] < 1.8][
                    'time'].min()

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
                    future_points = battery_data[
                        (battery_data['time'] >= battery_data.loc[first_stage_end]['time']) & (
                                    battery_data['time'] <= time_check)]
                    if len(future_points) > 1:
                        next_point = future_points.index[-1]
                    if (battery_data.loc[next_point]['time'] - battery_data.loc[first_stage_end][
                        'time']).total_seconds() >= 15 * 60:
                        if abs(battery_data.loc[next_point]['val'] - battery_data.loc[first_stage_end][
                            'val']) <= 0.0025:
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

        # 输入电压与电流数据
        '''clique_name,timestamp,val,time
        单体电池电压2V-001电池,1720540802,2.227,2024-07-09 16:00:02
        单体电池电压2V-001电池,1720540812,2.228,2024-07-09 16:00:12
         这是电压数据格式 clique_name从001一直到024                        这是电压数据格式

        clique_name,timestamp,val,time
         充放电电流,1720581055,0.699999,2024-07-10 03:10:55     这是电流数据格式
         
         使用Third Charge.csv作为电流数据文件
        '''

        stages_df = calculate_charge_stages(filtered_voltage_data, 'Third Charge.csv')
        print(stages_df)






    def calculate_cvct(data):
        """
        计算每节电池在恒压充电阶段的电压变化率平稳时长（CVCT）

        参数:
        data (DataFrame): 包含电压数据的DataFrame，必须包含 'timestamp', 'clique_name' 和 'val' 列。

        返回:
        dict: 包含每节电池CVCT值的字典，键是电池名称，值是电压变化率平稳时长（秒）。
        """

        # 设置打印选项，确保打印完整数组
        np.set_printoptions(threshold=np.inf)

        # 确保数据按时间戳排序
        data = data.sort_values(by='timestamp')

        # 获取唯一的电池名称
        batteries = [f'单体电池电压2V-{i:03d}电池' for i in range(1, 25)]

        # 创建一个字典来存储每节电池的CVCT值
        cvct_dict = {}

        # 处理每节电池的数据
        for battery in batteries:
            # 提取该电池的数据
            battery_data = data[data['clique_name'] == battery].reset_index(drop=True)

            # 确保有足够的数据点进行计算
            if len(battery_data) > 1:
                # 计算dV/dt
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
                    cvct = np.nan  # 如果没有平稳区域，返回NaN
            else:
                cvct = np.nan  # 如果数据点不足，返回NaN

            # 将结果存储在字典中
            cvct_dict[battery] = cvct

        return cvct_dict

    def calculate_max_dv_dt(data):
        """
        计算每节电池在恒压充电阶段的最大电压变化率（dV/dt）

        参数:
        data (DataFrame): 包含电压数据的DataFrame，必须包含 'timestamp', 'clique_name' 和 'val' 列

        返回:
        dict: 包含每节电池最大dV/dt值的字典，键是电池名称，值是最大电压变化率（dV/dt）
        """

        # 确保数据按时间戳排序
        data = data.sort_values(by='timestamp')

        # 获取唯一的电池名称
        batteries = [f'单体电池电压2V-{i:03d}电池' for i in range(1, 25)]

        # 存储每节电池的最大dV/dt值
        max_dv_dt_dict = {}

        # 处理每节电池的数据
        for battery in batteries:
            # 提取该电池的数据
            battery_data = data[data['clique_name'] == battery]

            # 确保有足够的数据点进行计算
            if len(battery_data) > 1:
                # 计算dV/dt
                time_diff = np.diff(battery_data['timestamp'])
                voltage_diff = np.diff(battery_data['val'])
                dv_dt = voltage_diff / time_diff

                # 找到最大dV/dt值
                max_dv_dt = np.max(dv_dt)
            else:
                max_dv_dt = np.nan  # 如果数据点不足，返回NaN

            # 将结果存储在字典中
            max_dv_dt_dict[battery] = max_dv_dt

        return max_dv_dt_dict

    def calculate_dtw(data, initial_time_cutoff, current_time_cutoff):
        """
        计算每节电池在两个时间段内的动态时间规整（DTW）距离。

        参数:
        data (DataFrame): 包含电压数据的DataFrame，必须包含 'time' 和 'val' 列。
        initial_time_cutoff (datetime): 第一个时间段的截止时间点。
        current_time_cutoff (datetime): 第二个时间段的起始时间点。

        返回:
        DataFrame: 包含每节电池DTW距离的DataFrame，列为 'clique_name' 和 'DTW_distance'。
        """

        # 确保数据按时间戳转换为datetime格式
        data['time'] = pd.to_datetime(data['time'])

        # 分割数据为初始时间段和当前时间段
        initial_curve = data[data['time'] <= initial_time_cutoff]
        current_curve = data[data['time'] >= current_time_cutoff]

        # 定义DTW计算函数
        def calculate_dtw(filtered_voltage_data):
            """
            计算每节电池在两个时间段内的动态时间规整（DTW）距离。

            参数:
            filtered_voltage_data (DataFrame): 包含电压数据的DataFrame，必须包含 'time' 和 'val' 列。

            返回:
            DataFrame: 包含每节电池DTW距离的DataFrame，列为 'clique_name' 和 'DTW_distance'。
            """

            # 分割数据为初始时间段和当前时间段
            initial_time_cutoff = filtered_voltage_data['DTW_wasserstein_initial_time_cutoff'].iloc[0]
            current_time_cutoff = filtered_voltage_data['DTW_wasserstein_current_time_cutoff'].iloc[0]
            initial_curve = filtered_voltage_data[filtered_voltage_data['time'] <= initial_time_cutoff]
            current_curve = filtered_voltage_data[filtered_voltage_data['time'] >= current_time_cutoff]

            # 定义DTW计算函数
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

    return pd.DataFrame(dtw_distances, columns=['clique_name', 'DTW_distance'])
    def calculate_wasserstein_distance(filtered_voltage_data):
        """
        计算每节电池两个时间段内的Wasserstein距离

        参数:
        filtered_voltage_data (DataFrame): 包含电压数据的DataFrame，必须包含 'time' 和 'val' 列。

        返回:
        DataFrame: 包含每节电池Wasserstein距离的DataFrame，列为 'clique_name' 和 'Wasserstein_distance'。
        """

        # 分割数据为初始时间段和当前时间段
        initial_time_cutoff = filtered_voltage_data['DTW_wasserstein_initial_time_cutoff'].iloc[0]
        current_time_cutoff = filtered_voltage_data['DTW_wasserstein_current_time_cutoff'].iloc[0]
        initial_curve = filtered_voltage_data[filtered_voltage_data['time'] <= initial_time_cutoff]
        current_curve = filtered_voltage_data[filtered_voltage_data['time'] >= current_time_cutoff]

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

        return pd.DataFrame(wasserstein_distances, columns=['clique_name', 'Wasserstein_distance'])

    def inner_function_vis():
        '''用来可视化电流和电压的数据，可以根据图像自行设置恒流/恒压的阈值'''
        voltage_file = '电压.csv'
        current_file = '电流.csv'

        voltage_data = pd.read_csv(voltage_file)
        current_data = pd.read_csv(current_file)

        # 将时间戳转换为日期时间格式
        voltage_data['time'] = pd.to_datetime(voltage_data['time'])
        current_data['time'] = pd.to_datetime(current_data['time'])

        # 创建图表
        fig, ax1 = plt.subplots(figsize=(15, 10))

        # 绘制电流数据
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Current (A)', color='tab:blue')
        ax1.plot(current_data['time'], current_data['val'], color='tab:blue', label='Current')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(which='both', linestyle='--', linewidth=0.5)

        # 创建第二个y轴来绘制电压数据
        ax2 = ax1.twinx()
        ax2.set_ylabel('Voltage (V)', color='tab:red')

        # 对每一个电池的电压数据进行绘制，并添加标签
        for column_name in voltage_data['clique_name'].unique():
            if column_name.startswith('单体电池电压2V-') and column_name.endswith('电池'):
                battery_data = voltage_data[voltage_data['clique_name'] == column_name]
                ax2.plot(battery_data['time'], battery_data['val'], label=column_name)

        ax2.tick_params(axis='y', labelcolor='tab:red')

        # 添加图例
        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # 主要刻度每0.1 V
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.02))  # 次要刻度每0.02 V

        # 设置电流数据的主要和次要刻度
        ax1.yaxis.set_major_locator(plt.MultipleLocator(1))  # 主要刻度每1 A
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.2))  #
        # 显示图表
        plt.title('Current and Voltage over Time')
        plt.show()

    # 数据可视化
    inner_function_vis()






    # 计算电压变化率平稳时长
    cvct_dict = calculate_cvct(filtered_voltage_data)

    # 打印每节电池的CVCT值
    for battery, cvct in cvct_dict.items():
        print(f"{battery}: 恒压充电电压变化率平稳时长 {cvct:.2f} 秒")

    # 计算最大充电电压变化率
    max_dv_dt_dict = calculate_max_dv_dt(filtered_voltage_data)

    # 打印每节电池的最大dV/dt值
    for battery, max_dv_dt in max_dv_dt_dict.items():
        print(f"{battery}: 最大充电电压变化率 {max_dv_dt:.6f} V/s")



    # 计算每个电池的DTW距离
    dtw_distances_df = calculate_dtw(data_voltage, initial_time_cutoff, current_time_cutoff)

    # 打印DTW距离结果
    print(dtw_distances_df)

    wasserstein_distances_df = calculate_wasserstein_distance(data_voltage, initial_time_cutoff, current_time_cutoff)

    # 打印Wasserstein距离结果
    print(wasserstein_distances_df)

# 调用主函数
main_function()
