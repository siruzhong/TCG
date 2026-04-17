"""Monthly total sunspot number (SILSO v2.0), January 1749 - present.

Single-variable monthly series. File columns (semicolon-separated):
    year ; month ; decimal_year ; SN ; std_dev ; num_obs ; definitive

We use only the monthly mean SN (column 3, 0-indexed).
"""
import json
import os

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

dataset_name = 'Sunspots'
raw_dir = base_dir + f'/datasets/raw_data/{dataset_name}'
raw_csv = raw_dir + '/SN_m_tot_V2.0.csv'
output_dir = base_dir + f'/datasets/{dataset_name}'
graph_file_path = None

frequency = 43200            # ~ minutes per month (30 * 24 * 60)
domain = 'solar activity'
timestamps_desc = ['time of day', 'day of week', 'day of month', 'day of year']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE'],
    'null_val': -1.0,
}


def load_and_preprocess_data():
    cols = ['year', 'month', 'decimal_year', 'sn', 'stddev', 'nobs', 'definitive']
    df = pd.read_csv(raw_csv, sep=';', names=cols, header=None)
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    df = df.set_index('date').sort_index()
    df.loc[df['sn'] < 0, 'sn'] = np.nan
    df['sn'] = df['sn'].interpolate().ffill().bfill()
    data = df[['sn']].astype(np.float32)
    print(f'Raw time series shape: {data.shape}')
    print(f'Period: {data.index[0].date()} -> {data.index[-1].date()}')
    return data


def add_temporal_features(df):
    l = df.shape[0]
    tod = np.zeros(l, dtype=np.float32)
    dow = np.zeros(l, dtype=np.float32)
    dom = np.zeros(l, dtype=np.float32)
    moy = (df.index.month.values - 1) / 12
    return np.stack([tod, dow, dom, moy], axis=-1)


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
