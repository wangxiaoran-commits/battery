import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import wasserstein_distance

def main_function():
    #数据前期处理
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

        # 保存
        voltage_data_total.to_csv('电压.csv', index=False)
        internal_resistance_data_total.to_csv('电阻.csv', index=False)

    def calculate_constant_voltage_duration(data, voltage_threshold=0.005):
        """
        计算每节电池的恒压充电时间

        参数:
        data (DataFrame): 包含电压数据的DataFrame，包含 'clique_name' 和 'val' 列
        voltage_threshold (float): 电压变化率的阈值，小于该阈值的电压视为恒压，阈值默认设置为0.005，在六个特征值函数定义之后的地方也可以更改阈值

        返回:
        dict: 包含每节电池恒压充电时长的字典，键是电池名称，值是恒压持续时间列表（秒）
        """

        def calculate_durations(battery_data):
            """
            计算单节电池在恒压充电阶段的持续时间

            参数:
            battery_data (DataFrame): 单节电池的数据，必须包含 'time' 和 'val' 列

            返回:
            list: 包含单节电池恒压充电时长的列表（秒）。
            """
            durations = []
            start_time = None
            end_time = None

            # 遍历电池数据，计算电压变化率
            for i in range(1, len(battery_data)):
                voltage_diff = abs(battery_data.iloc[i]['val'] - battery_data.iloc[i - 1]['val'])

                # 如果电压变化率小于等于阈值，则视为恒压状态
                if voltage_diff <= voltage_threshold:
                    if start_time is None:
                        start_time = pd.to_datetime(battery_data.iloc[i - 1]['time'])
                    end_time = pd.to_datetime(battery_data.iloc[i]['time'])
                else:
                    if start_time is not None and end_time is not None:
                        # 计算持续时间并记录
                        duration = (end_time - start_time).total_seconds()
                        durations.append(duration)
                    start_time = None
                    end_time = None

            # 处理最后一个持续时间段
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
                # 计算每节电池的恒压持续时间并记录
                durations = calculate_durations(battery_data)
                all_durations[battery_name] = durations

        return all_durations

    def calculate_constant_current_duration(data, current_threshold=2):
        """
        计算恒流充电时间

        参数:
        data (DataFrame): 包含电流数据的DataFrame，必须包含 'time' 和 'val' 列
        current_threshold (float): 电流变化率的阈值，小于该阈值的电流变化视为恒流，默认值为2，也可以自己设置

        返回:
        list: 包含恒流充电时长的列表（秒）。
        """

        start_time = None  # 记录恒流阶段的开始时间
        end_time = None  # 记录恒流阶段的结束时间
        durations = []  # 存储所有恒流阶段的持续时间

        # 遍历电流数据，计算电流变化率
        for i in range(1, len(data)):
            current_diff = abs(data.iloc[i]['val'] - data.iloc[i - 1]['val'])

            # 如果电流变化率小于等于阈值，则视为恒流状态
            if current_diff <= current_threshold:
                if start_time is None:
                    start_time = pd.to_datetime(data.iloc[i - 1]['time'])
                end_time = pd.to_datetime(data.iloc[i]['time'])
            else:
                if start_time is not None and end_time is not None:
                    # 计算持续时间并记录
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
        def dtw(A, B):
            """
            计算两个时间序列A和B之间的动态时间规整（DTW）距离

            参数:
            A (array): 第一个时间序列
            B (array): 第二个时间序列

            返回:
            float: 两个时间序列之间的DTW距离。
            """
            n, m = len(A), len(B)
            M = np.zeros((n, m))

            # 计算代价矩阵
            for i in range(n):
                for j in range(m):
                    M[i, j] = (A[i] - B[j]) ** 2

            # 初始化累积距离矩阵
            r = np.zeros((n, m))
            r[0, 0] = M[0, 0]

            # 动态规划填充累积距离矩阵
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

        # 计算每节电池的DTW距离
        for i in range(1, 25):
            battery_name = f'单体电池电压2V-{i:03d}电池'

            # 提取电池在两个时间段的数据
            initial_curve_battery = initial_curve[initial_curve['clique_name'] == battery_name]['val'].values
            current_curve_battery = current_curve[current_curve['clique_name'] == battery_name]['val'].values

            # 确保两个时间段都有数据点
            if len(initial_curve_battery) > 0 and len(current_curve_battery) > 0:
                # 计算DTW距离
                dtw_distance = dtw(initial_curve_battery, current_curve_battery)
                dtw_distances.append((battery_name, dtw_distance))
            else:
                dtw_distances.append((battery_name, None))

        # 返回包含DTW距离的DataFrame
        return pd.DataFrame(dtw_distances, columns=['clique_name', 'DTW_distance'])

    from scipy.stats import wasserstein_distance

    def calculate_wasserstein_distance(data, initial_time_cutoff, current_time_cutoff):
        """
        计算每节电池两个时间段内的Wasserstein距离

        参数:
        data (DataFrame): 包含电压数据的DataFrame，必须包含 'time' 和 'val' 列
        initial_time_cutoff (datetime): 第一个时间段的截止时间点
        current_time_cutoff (datetime): 第二个时间段的起始时间点

        返回:
        DataFrame: 包含每节电池Wasserstein距离的DataFrame，列为 'clique_name' 和 'Wasserstein_distance'。
        """

        # 确保数据按时间戳转换为datetime格式
        data['time'] = pd.to_datetime(data['time'])

        # 分割数据为初始时间段和当前时间段
        initial_curve = data[data['time'] <= initial_time_cutoff]
        current_curve = data[data['time'] >= current_time_cutoff]

        # 初始化存储Wasserstein距离的列表
        wasserstein_distances = []

        # 计算每节电池的Wasserstein距离
        for i in range(1, 25):
            battery_name = f'单体电池电压2V-{i:03d}电池'

            # 提取电池在两个时间段的数据
            initial_curve_battery = initial_curve[initial_curve['clique_name'] == battery_name]['val'].values
            current_curve_battery = current_curve[current_curve['clique_name'] == battery_name]['val'].values

            # 确保两个时间段都有数据点
            if len(initial_curve_battery) > 0 and len(current_curve_battery) > 0:
                # 计算Wasserstein距离
                distance = wasserstein_distance(initial_curve_battery, current_curve_battery)
                wasserstein_distances.append((battery_name, distance))
            else:
                wasserstein_distances.append((battery_name, None))

        # 返回包含Wasserstein距离的DataFrame
        return pd.DataFrame(wasserstein_distances, columns=['clique_name', 'Wasserstein_distance'])

    def inner_function_vis():
        '''用来可视化电流和电压的数据，可以根据图像自行设置恒流/恒压的阈值'''
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

    #数据可视化
    inner_function_vis()

    # 读取CSV文件
    data_voltage = pd.read_csv('电压.csv')
    data_current = pd.read_csv('充电电流.csv')

    # 确保数据按照时间排序
    data_current = data_current.sort_values(by='time')

    # 定义阈值
    voltage_threshold = 0.005 #恒压阈值
    current_threshold = 2  #恒流阈值

    # 计算所有电池的恒压充电时长
    all_voltage_durations = calculate_constant_voltage_duration(data_voltage, voltage_threshold)

    # 打印恒压充电时长结果
    for battery, durations in all_voltage_durations.items():
        print(f"{battery}, 恒压充电时间（秒）为 {durations}")

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

# Battery: 单体电池电压2V-001电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 10.0, 35750.0, 20.0, 189.0, 25150.0, 18836.0, 6467.0, 36178.0, 70.0, 690.0, 2693.0, 3440.0, 4523.0, 18342.0, 66220.0]
# Battery: 单体电池电压2V-002电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35760.0, 20.0, 189.0, 25150.0, 18836.0, 6467.0, 36168.0, 70.0, 29728.0, 66220.0]
# Battery: 单体电池电压2V-003电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35770.0, 20.0, 44195.0, 6478.0, 36178.0, 70.0, 6843.0, 22875.0, 66220.0]
# Battery: 单体电池电压2V-004电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35770.0, 20.0, 25349.0, 18836.0, 6467.0, 36178.0, 70.0, 3393.0, 7973.0, 18342.0, 66220.0]
# Battery: 单体电池电压2V-005电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35770.0, 20.0, 25349.0, 18836.0, 6478.0, 36168.0, 70.0, 29728.0, 66220.0]
# Battery: 单体电池电压2V-006电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35770.0, 20.0, 44195.0, 6478.0, 36178.0, 70.0, 29728.0, 66220.0]
# Battery: 单体电池电压2V-007电池, Constant Voltage Durations (in seconds): [40242.0, 19.0, 35770.0, 20.0, 25349.0, 18836.0, 6478.0, 36168.0, 70.0, 29728.0, 66220.0]
# Battery: 单体电池电压2V-008电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35780.0, 20.0, 25349.0, 25324.0, 36178.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-009电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35770.0, 20.0, 25349.0, 25324.0, 36168.0, 70.0, 3393.0, 26325.0, 66220.0]
# Battery: 单体电池电压2V-010电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35780.0, 20.0, 25349.0, 25324.0, 36178.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-011电池, Constant Voltage Durations (in seconds): [40242.0, 19.0, 35770.0, 20.0, 25349.0, 25324.0, 36178.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-012电池, Constant Voltage Durations (in seconds): [40242.0, 19.0, 35770.0, 20.0, 25349.0, 18836.0, 6478.0, 36178.0, 3473.0, 7973.0, 18342.0, 66220.0]
# Battery: 单体电池电压2V-013电池, Constant Voltage Durations (in seconds): [40242.0, 19.0, 35780.0, 20.0, 44205.0, 6478.0, 36178.0, 3473.0, 26325.0, 66220.0]
# Battery: 单体电池电压2V-014电池, Constant Voltage Durations (in seconds): [40242.0, 19.0, 35770.0, 20.0, 25359.0, 18836.0, 6478.0, 36178.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-015电池, Constant Voltage Durations (in seconds): [40252.0, 19.0, 35760.0, 20.0, 25359.0, 25324.0, 36168.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-016电池, Constant Voltage Durations (in seconds): [22499.0, 17743.0, 19.0, 35770.0, 20.0, 25359.0, 25324.0, 36168.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-017电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35780.0, 20.0, 25359.0, 25324.0, 36178.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-018电池, Constant Voltage Durations (in seconds): [22499.0, 17733.0, 19.0, 35770.0, 20.0, 50693.0, 36178.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-019电池, Constant Voltage Durations (in seconds): [22499.0, 17772.0, 10.0, 35739.0, 20.0, 25359.0, 25324.0, 36178.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-020电池, Constant Voltage Durations (in seconds): [22499.0, 17743.0, 19.0, 35760.0, 10.0, 25359.0, 25324.0, 36168.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-021电池, Constant Voltage Durations (in seconds): [22499.0, 17743.0, 19.0, 35760.0, 20.0, 25359.0, 25324.0, 36178.0, 6923.0, 22875.0, 66220.0]
# Battery: 单体电池电压2V-022电池, Constant Voltage Durations (in seconds): [22499.0, 17743.0, 19.0, 10.0, 35750.0, 20.0, 25359.0, 25324.0, 36168.0, 3473.0, 26325.0, 66220.0]
# Battery: 单体电池电压2V-023电池, Constant Voltage Durations (in seconds): [22499.0, 17743.0, 19.0, 35770.0, 20.0, 25359.0, 25324.0, 36178.0, 29808.0, 66220.0]
# Battery: 单体电池电压2V-024电池, Constant Voltage Durations (in seconds): [22499.0, 17743.0, 19.0, 10.0, 35750.0, 20.0, 25359.0, 25324.0, 36178.0, 6923.0, 22875.0, 66220.0]
# 恒流充电时长 1: 20.0 秒
# 恒流充电时长 2: 1279.0 秒
# 恒流充电时长 3: 568.0 秒
# 恒流充电时长 4: 1219.0 秒
# 恒流充电时长 5: 797.0 秒
# 恒流充电时长 6: 506.0 秒
# 恒流充电时长 7: 939.0 秒
# 恒流充电时长 8: 652.0 秒
# 恒流充电时长 9: 1782.0 秒
# 恒流充电时长 10: 17519.0 秒
# 恒流充电时长 11: 25325.0 秒
# 恒流充电时长 12: 50.0 秒
# 恒流充电时长 13: 691.0 秒
# 恒流充电时长 14: 900.0 秒
# 恒流充电时长 15: 1783.0 秒
# 恒流充电时长 16: 911.0 秒
# 恒流充电时长 17: 650.0 秒
# 恒流充电时长 18: 441.0 秒
# 恒流充电时长 19: 819.0 秒
# 恒流充电时长 20: 580.0 秒
# 恒流充电时长 21: 2492.0 秒
# 恒流充电时长 22: 2020.0 秒
# 恒流充电时长 23: 18343.0 秒
# 恒流充电时长 24: 17137.0 秒
# 恒流充电时长 25: 49062.0 秒
# 单体电池电压2V-001电池: 恒压充电电压变化率平稳时长 21428.00 秒
# 单体电池电压2V-002电池: 恒压充电电压变化率平稳时长 22499.00 秒
# 单体电池电压2V-003电池: 恒压充电电压变化率平稳时长 20103.00 秒
# 单体电池电压2V-004电池: 恒压充电电压变化率平稳时长 23208.00 秒
# 单体电池电压2V-005电池: 恒压充电电压变化率平稳时长 16934.00 秒
# 单体电池电压2V-006电池: 恒压充电电压变化率平稳时长 33662.00 秒
# 单体电池电压2V-007电池: 恒压充电电压变化率平稳时长 22499.00 秒
# 单体电池电压2V-008电池: 恒压充电电压变化率平稳时长 17777.00 秒
# 单体电池电压2V-009电池: 恒压充电电压变化率平稳时长 22315.00 秒
# 单体电池电压2V-010电池: 恒压充电电压变化率平稳时长 22499.00 秒
# 单体电池电压2V-011电池: 恒压充电电压变化率平稳时长 24387.00 秒
# 单体电池电压2V-012电池: 恒压充电电压变化率平稳时长 26385.00 秒
# 单体电池电压2V-013电池: 恒压充电电压变化率平稳时长 20660.00 秒
# 单体电池电压2V-014电池: 恒压充电电压变化率平稳时长 33918.00 秒
# 单体电池电压2V-015电池: 恒压充电电压变化率平稳时长 40242.00 秒
# 单体电池电压2V-016电池: 恒压充电电压变化率平稳时长 19421.00 秒
# 单体电池电压2V-017电池: 恒压充电电压变化率平稳时长 23136.00 秒
# 单体电池电压2V-018电池: 恒压充电电压变化率平稳时长 27711.00 秒
# 单体电池电压2V-019电池: 恒压充电电压变化率平稳时长 22499.00 秒
# 单体电池电压2V-020电池: 恒压充电电压变化率平稳时长 22499.00 秒
# 单体电池电压2V-021电池: 恒压充电电压变化率平稳时长 18237.00 秒
# 单体电池电压2V-022电池: 恒压充电电压变化率平稳时长 15860.00 秒
# 单体电池电压2V-023电池: 恒压充电电压变化率平稳时长 25559.00 秒
# 单体电池电压2V-024电池: 恒压充电电压变化率平稳时长 25894.00 秒
# 单体电池电压2V-001电池: 最大充电电压变化率 0.008600 V/s
# 单体电池电压2V-002电池: 最大充电电压变化率 0.007600 V/s
# 单体电池电压2V-003电池: 最大充电电压变化率 0.006700 V/s
# 单体电池电压2V-004电池: 最大充电电压变化率 0.007100 V/s
# 单体电池电压2V-005电池: 最大充电电压变化率 0.007400 V/s
# 单体电池电压2V-006电池: 最大充电电压变化率 0.007700 V/s
# 单体电池电压2V-007电池: 最大充电电压变化率 0.007900 V/s
# 单体电池电压2V-008电池: 最大充电电压变化率 0.007200 V/s
# 单体电池电压2V-009电池: 最大充电电压变化率 0.007700 V/s
# 单体电池电压2V-010电池: 最大充电电压变化率 0.007600 V/s
# 单体电池电压2V-011电池: 最大充电电压变化率 0.007800 V/s
# 单体电池电压2V-012电池: 最大充电电压变化率 0.009100 V/s
# 单体电池电压2V-013电池: 最大充电电压变化率 0.007600 V/s
# 单体电池电压2V-014电池: 最大充电电压变化率 0.007900 V/s
# 单体电池电压2V-015电池: 最大充电电压变化率 0.008500 V/s
# 单体电池电压2V-016电池: 最大充电电压变化率 0.008500 V/s
# 单体电池电压2V-017电池: 最大充电电压变化率 0.008000 V/s
# 单体电池电压2V-018电池: 最大充电电压变化率 0.008700 V/s
# 单体电池电压2V-019电池: 最大充电电压变化率 0.008600 V/s
# 单体电池电压2V-020电池: 最大充电电压变化率 0.008900 V/s
# 单体电池电压2V-021电池: 最大充电电压变化率 0.009000 V/s
# 单体电池电压2V-022电池: 最大充电电压变化率 0.009000 V/s
# 单体电池电压2V-023电池: 最大充电电压变化率 0.008000 V/s
# 单体电池电压2V-024电池: 最大充电电压变化率 0.009400 V/s
#        clique_name  DTW_distance
# 0   单体电池电压2V-001电池    291.360893
# 1   单体电池电压2V-002电池    272.927781
# 2   单体电池电压2V-003电池    254.656690
# 3   单体电池电压2V-004电池    259.752499
# 4   单体电池电压2V-005电池    261.919290
# 5   单体电池电压2V-006电池    271.652413
# 6   单体电池电压2V-007电池    268.627989
# 7   单体电池电压2V-008电池    257.033780
# 8   单体电池电压2V-009电池    260.055251
# 9   单体电池电压2V-010电池    256.990933
# 10  单体电池电压2V-011电池    262.263512
# 11  单体电池电压2V-012电池    290.348166
# 12  单体电池电压2V-013电池    253.643723
# 13  单体电池电压2V-014电池    255.577748
# 14  单体电池电压2V-015电池    262.195546
# 15  单体电池电压2V-016电池    260.445357
# 16  单体电池电压2V-017电池    260.892049
# 17  单体电池电压2V-018电池    270.346560
# 18  单体电池电压2V-019电池    266.429302
# 19  单体电池电压2V-020电池    268.743638
# 20  单体电池电压2V-021电池    268.030459
# 21  单体电池电压2V-022电池    269.192334
# 22  单体电池电压2V-023电池    260.589392
# 23  单体电池电压2V-024电池    271.558338
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


