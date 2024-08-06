import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import time

def main_function():
    main_file_path = r"C:\Users\amber\Desktop\特征蓄电池吃\battery_0710-0712.csv"

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

        voltage_data_total.to_csv('电压.csv', index=False)
        current_data_total.to_csv('电流.csv', index=False)

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

        filtered_voltage_data['DTW_wasserstein_initial_time_cutoff'] = datetime.strptime('2024-07-10 13:09', '%Y-%m-%d %H:%M')
        filtered_voltage_data['DTW_wasserstein_current_time_cutoff'] = datetime.strptime('2024-07-11 13:20', '%Y-%m-%d %H:%M')
        filtered_voltage_data.to_csv('filtered_voltage_data.csv', index=False)

        initial_time_cutoff = datetime.strptime('2024-07-10 13:09', '%Y-%m-%d %H:%M')
        current_time_cutoff = datetime.strptime('2024-07-11 13:20', '%Y-%m-%d %H:%M')

        initial_curve = data[(data['time'] >= initial_time_cutoff) & (data['time'] <= datetime.strptime('2024-07-11 11:14', '%Y-%m-%d %H:%M'))]
        current_curve = data[data['time'] >= current_time_cutoff]

        return filtered_voltage_data, initial_curve, current_curve

    start_time_str = '2024-07-11 13:20'
    duration_hours = 27
    filtered_voltage_data, initial_curve, current_curve = process_data('电压.csv', '电流.csv', start_time_str, duration_hours)

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

    dtw_distances_df = calculate_dtw(initial_curve, current_curve)
    print(dtw_distances_df)

main_function()
