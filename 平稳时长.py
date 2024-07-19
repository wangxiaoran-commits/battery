import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)

file_path = '电压.csv'
data = pd.read_csv(file_path)

# 确保数据按时间戳排序
data = data.sort_values(by='timestamp')

# 获取唯一的电池名称
batteries = [f'单体电池电压2V-{i:03d}电池' for i in range(1, 25)]

# 创建一个字典来存储每节电池的CVTCT值
cvtct_dict = {}

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

        # 打印中间数据以便调试
        print(f"{battery} 电压变化率: {dv_dt}")

        # 找到平稳区域（-0.0001 <= dV/dt <= 0.0002）
        stable_region = (dv_dt >= -0.0001) & (dv_dt <= 0.0002)

        # 打印平稳区域判断结果以便调试
        print(f"{battery} 平稳区域标记: {stable_region}")

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

        # 打印所有的平稳区域持续时间以便调试
        print(f"{battery} 平稳区域持续时间: {stable_durations}")

        # 选取最长的平稳区域持续时间
        if stable_durations:
            cvtct = max(stable_durations)
        else:
            cvtct = np.nan  # 如果没有平稳区域，返回NaN
    else:
        cvtct = np.nan  # 如果数据点不足，返回NaN

    # 将结果存储在字典中
    cvtct_dict[battery] = cvtct

# 打印每节电池的CVTCT值
for battery, cvtct in cvtct_dict.items():
    print(f"{battery}: 恒压充电电压变化率平稳时长 {cvtct:.2f} 秒")

# 结果保存为DataFrame并展示
result_df = pd.DataFrame(list(cvtct_dict.items()), columns=['Battery', 'CVTCT (s)'])

#
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