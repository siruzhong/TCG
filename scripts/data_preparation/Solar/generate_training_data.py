import json
import os

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# Hyperparameters
dataset_name = 'Solar'
data_file_path = base_dir + f'/datasets/raw_data/{dataset_name}/solar_AL.txt'
graph_file_path = None
output_dir = base_dir + f'/datasets/{dataset_name}'
add_time_of_day = True
add_day_of_week = True
add_day_of_month = True
add_day_of_year = True
steps_per_day = 144  # 10-minute frequency: 1440 / 10
frequency = 1440 // steps_per_day
start_date = '2006-01-01 00:00:00'
domain = 'solar power plants'
timestamps_desc = ['time of day', 'day of week', 'day of month', 'day of year']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE'],
    'null_val': np.nan,
}


def load_and_preprocess_data():
    '''Load raw Solar-Energy (LSTNet) data; 52560 x 137 matrix of 10-min readings.'''
    data = np.loadtxt(data_file_path, delimiter=',', dtype=np.float32)
    print(f'Raw time series shape: {data.shape}')
    return data


def add_temporal_features(data):
    '''Add time of day, day of week, day of month, and day of year as features.'''
    l = data.shape[0]
    index = pd.date_range(start=start_date, periods=l, freq=f'{frequency}min')
    timestamps = []

    if add_time_of_day:
        tod = np.array([i % steps_per_day / steps_per_day for i in range(l)])
        timestamps.append(tod)

    if add_day_of_week:
        dow = index.dayofweek.values / 7
        timestamps.append(dow)

    if add_day_of_month:
        dom = (index.day.values - 1) / 31
        timestamps.append(dom)

    if add_day_of_year:
        doy = (index.dayofyear.values - 1) / 366
        timestamps.append(doy)

    timestamps = np.stack(timestamps, axis=-1)
    return timestamps


def split_and_save_data(data, timestamps):
    '''Split the dataset by ratio and persist each split as a .npy file.'''
    train_ratio, val_ratio, _ = regular_settings['train_val_test_ratio']
    train_len = int(data.shape[0] * train_ratio)
    val_len = int(data.shape[0] * val_ratio)

    train_data = data[:train_len].astype(np.float32)
    val_data = data[train_len: train_len + val_len].astype(np.float32)
    test_data = data[train_len + val_len:].astype(np.float32)
    train_timestamps = timestamps[:train_len].astype(np.float32)
    val_timestamps = timestamps[train_len: train_len + val_len].astype(np.float32)
    test_timestamps = timestamps[train_len + val_len:].astype(np.float32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(output_dir, 'train_timestamps.npy'), train_timestamps)
    np.save(os.path.join(output_dir, 'val_timestamps.npy'), val_timestamps)
    np.save(os.path.join(output_dir, 'test_timestamps.npy'), test_timestamps)
    print(f'train_data shape: {train_data.shape}')
    print(f'val_data shape:   {val_data.shape}')
    print(f'test_data shape:  {test_data.shape}')
    print(f'Data saved to {output_dir}')


def save_description(data, timestamps):
    '''Write dataset metadata to meta.json for downstream tooling.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'frequency (minutes)': frequency,
        'shape': list(data.shape),
        'timestamps_shape': list(timestamps.shape),
        'timestamps_description': timestamps_desc,
        'num_time_steps': data.shape[0],
        'num_vars': data.shape[1],
        'has_graph': graph_file_path is not None,
        'regular_settings': regular_settings,
    }
    description_path = os.path.join(output_dir, 'meta.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4, default=str)
    print(f'Description saved to {description_path}')


def main():
    print(f'---------- Generating {dataset_name} data ----------')
    data = load_and_preprocess_data()
    timestamps = add_temporal_features(data)
    split_and_save_data(data, timestamps)
    save_description(data, timestamps)


if __name__ == '__main__':
    main()
