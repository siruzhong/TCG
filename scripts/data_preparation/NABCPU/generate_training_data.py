"""NAB realAWSCloudwatch EC2 CPU utilization dataset.

The NAB repository provides seven EC2 CPU traces. Three of them span the
identical window 2014-02-14 ~ 2014-02-28, the rest span 2014-04-02 ~
2014-04-24. We keep the three Feb-2014 machines that overlap perfectly,
resample each onto a 5-minute grid, and take the intersection -> a clean
aligned multivariate ~(4k, 3) dataset of server CPU utilisation with sharp
step-changes (scheduled jobs / bursts).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

dataset_name = 'NABCPU'
raw_dir = base_dir + f'/datasets/raw_data/{dataset_name}'
output_dir = base_dir + f'/datasets/{dataset_name}'
graph_file_path = None

steps_per_day = 288          # 5-min frequency
frequency = 5
domain = 'cloud CPU utilization'
timestamps_desc = ['time of day', 'day of week', 'day of month', 'day of year']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE'],
    'null_val': np.nan,
}


ALIGNED_MACHINES = ['24ae8d', '5f5533', 'fe7f93']


def load_and_preprocess_data():
    series = {}
    for mid in ALIGNED_MACHINES:
        f = os.path.join(raw_dir, f'ec2_cpu_utilization_{mid}.csv')
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('5min').mean().interpolate()
        series[mid] = df['value']

    combined = pd.concat(series.values(), axis=1, keys=series.keys())
    start = max(s.index[0] for s in series.values())
    end = min(s.index[-1] for s in series.values())
    combined = combined.loc[start:end].dropna(how='any').astype(np.float32)
    print(f'Raw time series shape: {combined.shape}')
    print(f'Period: {combined.index[0]} -> {combined.index[-1]}')
    return combined


def add_temporal_features(df):
    l = df.shape[0]
    tod = (df.index.hour.values * 60 + df.index.minute.values) / (24 * 60)
    dow = df.index.dayofweek.values / 7
    dom = (df.index.day.values - 1) / 31
    doy = (df.index.dayofyear.values - 1) / 366
    return np.stack([tod, dow, dom, doy], axis=-1)


def split_and_save_data(data, timestamps):
    train_ratio, val_ratio, _ = regular_settings['train_val_test_ratio']
    train_len = int(data.shape[0] * train_ratio)
    val_len = int(data.shape[0] * val_ratio)
    os.makedirs(output_dir, exist_ok=True)
    for name, start, end in [('train', 0, train_len),
                             ('val',   train_len, train_len + val_len),
                             ('test',  train_len + val_len, data.shape[0])]:
        arr = data[start:end].astype(np.float32)
        ts = timestamps[start:end].astype(np.float32)
        np.save(os.path.join(output_dir, f'{name}_data.npy'), arr)
        np.save(os.path.join(output_dir, f'{name}_timestamps.npy'), ts)
        print(f'{name}_data shape: {arr.shape}')
    print(f'Data saved to {output_dir}')


def save_description(data, timestamps):
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
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(description, f, indent=4, default=str)


def main():
    print(f'---------- Generating {dataset_name} data ----------')
    df = load_and_preprocess_data()
    ts = add_temporal_features(df)
    split_and_save_data(df.values, ts)
    save_description(df.values, ts)


if __name__ == '__main__':
    main()
