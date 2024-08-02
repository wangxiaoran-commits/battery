import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 读取所有电压数据
data = pd.read_csv('电压.csv')

# 将time列转换为datetime格式
data['time'] = pd.to_datetime(data['time'])

# 定义时间分割点
current_time_cutoff = datetime.strptime('2024-07-11 13:20', '%Y-%m-%d %H:%M')

# 过滤出时间大于等于current_time_cutoff的数据
filtered_data = data[data['time'] >= current_time_cutoff]

# 设置图形大小
plt.figure(figsize=(15, 10))

# 可视化每节电池的电压数据
for i in range(1, 25):
    battery_name = f'单体电池电压2V-{i:03d}电池'
    battery_data = filtered_data[filtered_data['clique_name'] == battery_name]

    # 绘制电压曲线
    plt.plot(battery_data['time'], battery_data['val'], label=battery_name)

# 设置图形的标题和标签
plt.title('Voltage Curves for 24 Batteries after 2024-07-11 13:20')
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

# 调整刻度
plt.xticks(rotation=45)
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
