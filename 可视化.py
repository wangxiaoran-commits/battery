import pandas as pd
import matplotlib.pyplot as plt

# 读取电压和电流数据
voltage_file = '电压.csv'
current_file = '充电电流.csv'

voltage_data = pd.read_csv(voltage_file)
current_data = pd.read_csv(current_file)

# 将时间戳转换为日期时间格式
voltage_data['time'] = pd.to_datetime(voltage_data['time'])
current_data['time'] = pd.to_datetime(current_data['time'])

# 创建一个图表来显示电流和电压随时间的变化
fig, ax1 = plt.subplots(figsize=(15, 10))

# 绘制电流数据
ax1.set_xlabel('Time')
ax1.set_ylabel('Current (A)', color='tab:blue')
ax1.plot(current_data['time'], current_data['val'], color='tab:blue', label='Current')
ax1.tick_params(axis='y', labelcolor='tab:blue')

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

# 显示图表
plt.title('Current and Voltage over Time')
plt.show()

