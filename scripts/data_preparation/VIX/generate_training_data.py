"""VIX (CBOE Volatility Index) daily close, downloaded from FRED.

Univariate daily series from 1990 onwards. Features the classic
``volatility clustering`` regime structure: long quiet stretches at low
levels punctuated by sharp spikes during market stress.
"""
import json
import os

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

dataset_name = 'VIX'
raw_dir = base_dir + f'/datasets/raw_data/{dataset_name}'
raw_csv = raw_dir + '/VIXCLS.csv'
output_dir = base_dir + f'/datasets/{dataset_name}'
graph_file_path = None

frequency = 1440
domain = 'market volatility'
timestamps_desc = ['time of day', 'day of week', 'day of month', 'day of year']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE'],
    'null_val': np.nan,
}


def load_and_preprocess_data():
    df = pd.read_csv(raw_csv)
    df.columns = [c.strip() for c in df.columns]
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.set_index('observation_date').sort_index()
    df = df.replace('.', np.nan)
    df['VIXCLS'] = pd.to_numeric(df['VIXCLS'], errors='coerce')
    df = df.dropna().astype(np.float32)
    print(f'Raw time series shape: {df.shape}')
    print(f'Period: {df.index[0].date()} -> {df.index[-1].date()}')
    return df


def add_temporal_features(df):
    l = df.shape[0]
    timestamps = []
    timestamps.append(np.zeros(l, dtype=np.float32))
    timestamps.append(df.index.dayofweek.values / 7)
    timestamps.append((df.index.day.values - 1) / 31)
    timestamps.append((df.index.dayofyear.values - 1) / 366)
    return np.stack(timestamps, axis=-1)


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
