import pandas as pd
import os

# 文件路径
file_path = r"C:\Users\amber\Desktop\特征蓄电池吃\battery_0710-0712.csv"  # 替换成你的CSV文件路径

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # 定义处理每个chunk的函数
    def process_chunk(chunk):
        # 将 'timestamp' 从 Unix 时间转换为 datetime
        chunk['time'] = pd.to_datetime(chunk['timestamp'], unit='s')

        # 过滤并分类数据
        voltage_data = chunk[chunk['clique_name'].str.contains('单体电池电压2V-')]
        internal_resistance_data = chunk[chunk['clique_name'].str.contains('单体电池电压2V内阻-')]

        return voltage_data, internal_resistance_data

    # 初始化空DataFrame来存储处理后的数据
    voltage_data_total = pd.DataFrame()
    internal_resistance_data_total = pd.DataFrame()

    # 分块读取和处理CSV文件
    chunk_size = 100000  # 根据你的可用内存调整chunk大小

    # 使用read_csv函数分块读取
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        voltage_data_chunk, internal_resistance_data_chunk = process_chunk(chunk)

        # 将处理后的chunk附加到总数据中
        voltage_data_total = pd.concat([voltage_data_total, voltage_data_chunk], ignore_index=True)
        internal_resistance_data_total = pd.concat([internal_resistance_data_total, internal_resistance_data_chunk], ignore_index=True)

    # 保存处理后的数据到新的CSV文件
    voltage_data_total.to_csv('电压.csv', index=False)
    internal_resistance_data_total.to_csv('电阻.csv', index=False)

    print("Data processing complete. Files saved as 'voltage_data_total.csv' and 'internal_resistance_data_total.csv'.")
# import pandas as pd
# import os
#
# # 文件路径
# file_path = r"C:\Users\amber\Desktop\特征蓄电池吃\battery_0710-0712.csv"  # 替换成你的CSV文件路径
#
# # 检查文件是否存在
# if not os.path.exists(file_path):
#     print(f"File not found: {file_path}")
# else:
#     # 定义处理每个chunk的函数
#     def process_chunk(chunk):
#         # 将 'timestamp' 从 Unix 时间转换为 datetime
#         chunk['time'] = pd.to_datetime(chunk['timestamp'], unit='s')
#
#         # 过滤出变量名为 '充放电电流' 的数据
#         current_data = chunk[chunk['clique_name'] == '充放电电流']
#
#         return current_data
#
#     # 初始化空DataFrame来存储处理后的数据
#     current_data_total = pd.DataFrame()
#
#     # 分块读取和处理CSV文件
#     chunk_size = 100000  # 根据你的可用内存调整chunk大小
#
#     # 使用read_csv函数分块读取
#     for chunk in pd.read_csv(file_path, chunksize=chunk_size):
#         current_data_chunk = process_chunk(chunk)
#
#         # 将处理后的chunk附加到总数据中
#         current_data_total = pd.concat([current_data_total, current_data_chunk], ignore_index=True)
#
#     # 保存处理后的数据到新的CSV文件
#     current_data_total.to_csv('充电电流.csv', index=False)
#
#     print("Data processing complete. File saved as '充电电流.csv'")