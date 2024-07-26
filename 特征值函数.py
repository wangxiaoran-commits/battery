import os
import pandas as pd
import matplotlib.pyplot as plt

import os
import pandas as pd
import matplotlib.pyplot as plt

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
            durations = calculate_durations(battery_data)
            all_durations[battery_name] = durations

        return all_durations

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
    data = pd.read_csv('电压.csv')

    # 定义电压阈值
    voltage_threshold = 0.2

    # 计算所有电池的恒压充电时长
    all_durations = calculate_constant_voltage_duration(data, voltage_threshold)

    # 打印结果
    for battery, durations in all_durations.items():
        print(f"Battery: {battery}, Constant Voltage Durations (in seconds): {durations}")

# 调用主函数
main_function()



