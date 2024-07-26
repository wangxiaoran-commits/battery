import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import wasserstein_distance

def main_function():
    # 文件路径
    file_path = r"C:\Users\amber\Desktop\特征蓄电池吃\battery_0710-0712.csv"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    else:
        # 定义处理每个chunk的函数
        def process_chunk(chunk):
            # 将 'timestamp' 从 Unix 时间转换为 datetime
            chunk['time'] = pd.to_datetime(chunk['timestamp'], unit='s')

            # 过滤并分类数据
            voltage_data = chunk[chunk['clique_name'].str.contains('单体电池电压2V-')]
            internal_resistance_data = chunk[chunk['clique_name'].str.contains('单体电池电压2V内阻-')]

            return voltage_data, internal_resistance_data

        # 新建DataFrame
        voltage_data_total = pd.DataFrame()
        internal_resistance_data_total = pd.DataFrame()

        # 分块读取和处理
        chunk_size = 100000

        # 分块读取
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            voltage_data_chunk, internal_resistance_data_chunk = process_chunk(chunk)

            # 处理后的chunk加到总数据
            voltage_data_total = pd.concat([voltage_data_total, voltage_data_chunk], ignore_index=True)
            internal_resistance_data_total = pd.concat([internal_resistance_data_total, internal_resistance_data_chunk],
                                                       ignore_index=True)

        # 保存新数据
        voltage_data_total.to_csv('电压.csv', index=False)
        internal_resistance_data_total.to_csv('电阻.csv', index=False)

    def calculate_constant_voltage_duration(data, voltage_threshold):
        def calculate_durations(battery_data):
            durations = []
            start_time = None
            end_time = None

            for i in range(1, len(battery_data)):
                voltage_diff = abs(battery_data.iloc[i]['val'] - battery_data.iloc[i - 1]['val'])

                if voltage_diff <= voltage_threshold:
                    if start_time is None:
                        start_time = pd.to_datetime(battery_data.iloc[i - 1]['time'])
                    end_time = pd.to_datetime(battery_data.iloc[i]['time'])
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

        # 存储所有电池的恒压充电时长
        all_durations = {}

        # 循环处理每节电池
        for i in range(1, 25):
            battery_name = f'单体电池电压2V-{i:03d}电池'
            battery_data = data[data['clique_name'] == battery_name]
            if battery_data.empty:
                print(f"No data for {battery_name}")
            else:
                durations = calculate_durations(battery_data)
                all_durations[battery_name] = durations

        return all_durations

    def calculate_constant_current_duration(data, current_threshold=2):
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

    def calculate_cvct(data):
        np.set_printoptions(threshold=np.inf)

        # 确保数据按时间戳排序
        data = data.sort_values(by='timestamp')

        # 获取唯一的电池名称
        batteries = [f'单体电池电压2V-{i:03d}电池' for i in range(1, 25)]

        # 创建一个字典来存储每节电池的CVTCT值
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
        # 确保数据按时间戳排序
        data = data.sort_values(by='timestamp')

        # 获取唯一的电池名称
        batteries = [f'单体电池电压2V-{i:03d}电池' for i in range(1, 25)]

        # 创建一个字典来存储每节电池的最大dV/dt值
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
        # 确保数据按时间戳转换为datetime格式
        data['time'] = pd.to_datetime(data['time'])

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

    def calculate_wasserstein_distance(data, initial_time_cutoff, current_time_cutoff):
        # 确保数据按时间戳转换为datetime格式
        data['time'] = pd.to_datetime(data['time'])

        # 分割数据
        initial_curve = data[data['time'] <= initial_time_cutoff]
        current_curve = data[data['time'] >= current_time_cutoff]

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

        return pd.DataFrame(wasserstein_distances, columns=['clique_name', 'Wasserstein_distance'])

    def inner_function_vis():
        voltage_file = '电压.csv'
        current_file = '充电电流.csv'

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

    # 调用内嵌函数
    inner_function_vis()

    # 读取CSV文件
    data_voltage = pd.read_csv('电压.csv')
    data_current = pd.read_csv('充电电流.csv')

    # 确保数据按照时间排序
    data_current = data_current.sort_values(by='time')

    # 定义阈值
    voltage_threshold = 0.005
    current_threshold = 2

    # 计算所有电池的恒压充电时长
    all_voltage_durations = calculate_constant_voltage_duration(data_voltage, voltage_threshold)

    # 打印恒压充电时长结果
    for battery, durations in all_voltage_durations.items():
        print(f"Battery: {battery}, Constant Voltage Durations (in seconds): {durations}")

    # 计算恒流充电时长
    constant_current_durations = calculate_constant_current_duration(data_current, current_threshold)

    # 输出恒流充电时长结果
    for idx, duration in enumerate(constant_current_durations):
        print(f"恒流充电时长 {idx + 1}: {duration} 秒")

    # total_constant_current_duration = sum(constant_current_durations)
    # print(f"总的恒流充电时间: {total_constant_current_duration} 秒")

    # 计算电压变化率平稳时长
    cvct_dict = calculate_cvct(data_voltage)

    # 打印每节电池的CVCT值
    for battery, cvct in cvct_dict.items():
        print(f"{battery}: 恒压充电电压变化率平稳时长 {cvct:.2f} 秒")

    # 计算最大充电电压变化率
    max_dv_dt_dict = calculate_max_dv_dt(data_voltage)

    # 打印每节电池的最大dV/dt值
    for battery, max_dv_dt in max_dv_dt_dict.items():
        print(f"{battery}: 最大充电电压变化率 {max_dv_dt:.6f} V/s")

    # 定义时间分割点
    initial_time_cutoff = datetime.strptime('2024-07-11 11:15', '%Y-%m-%d %H:%M')
    current_time_cutoff = datetime.strptime('2024-07-11 21:20', '%Y-%m-%d %H:%M')

    # 计算每个电池的DTW距离
    dtw_distances_df = calculate_dtw(data_voltage, initial_time_cutoff, current_time_cutoff)

    # 打印DTW距离结果
    print(dtw_distances_df)

    wasserstein_distances_df = calculate_wasserstein_distance(data_voltage, initial_time_cutoff, current_time_cutoff)

    # 打印Wasserstein距离结果
    print(wasserstein_distances_df)

# 调用主函数
main_function()




