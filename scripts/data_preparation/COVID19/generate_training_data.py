"""COVID-19 daily new-cases dataset from Johns Hopkins CSSE.

Produces a multivariate daily series (T, C) where each channel is the per-day
new case count of one country. We pick the top-C countries by cumulative
cases plus a "Global" aggregate so that the dataset has a clear ``step-change
+ exponential burst`` regime structure (see appendix).
"""
import json
import os

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

dataset_name = 'COVID19'
raw_dir = base_dir + f'/datasets/raw_data/{dataset_name}'
confirmed_path = raw_dir + '/time_series_covid19_confirmed_global.csv'
output_dir = base_dir + f'/datasets/{dataset_name}'
graph_file_path = None

N_TOP_COUNTRIES = 7
steps_per_day = 1           # daily data
frequency = 1440            # minutes per sample (1 day)
domain = 'COVID-19 new cases'
timestamps_desc = ['time of day', 'day of week', 'day of month', 'day of year']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE'],
    'null_val': np.nan,
}


def load_and_preprocess_data():
    df = pd.read_csv(confirmed_path)
    df = df.drop(columns=['Province/State', 'Lat', 'Long'])
    df = df.groupby('Country/Region').sum()
    cumulative = df.T
    cumulative.index = pd.to_datetime(cumulative.index, format='%m/%d/%y')
    cumulative = cumulative.sort_index()

    totals = cumulative.iloc[-1].sort_values(ascending=False)
    top_countries = totals.head(N_TOP_COUNTRIES).index.tolist()
    selected = cumulative[top_countries].copy()
    selected['Global'] = cumulative.sum(axis=1)

    daily_new = selected.diff().fillna(0.0)
    daily_new[daily_new < 0] = 0.0

    print(f'Raw time series shape: {daily_new.shape}')
    print('Channels:', list(daily_new.columns))
    return daily_new


def add_temporal_features(df):
    l = df.shape[0]
    timestamps = []
    tod = np.zeros(l, dtype=np.float32)
    timestamps.append(tod)
    dow = df.index.dayofweek.values / 7
    timestamps.append(dow)
    dom = (df.index.day.values - 1) / 31
    timestamps.append(dom)
    doy = (df.index.dayofyear.values - 1) / 366
    timestamps.append(doy)
    return np.stack(timestamps, axis=-1)


def split_and_save_data(data, timestamps):
    train_ratio, val_ratio, _ = regular_settings['train_val_test_ratio']
    train_len = int(data.shape[0] * train_ratio)
    val_len = int(data.shape[0] * val_ratio)

    splits = [('train', 0, train_len),
              ('val',   train_len, train_len + val_len),
              ('test',  train_len + val_len, data.shape[0])]

    os.makedirs(output_dir, exist_ok=True)
    for name, start, end in splits:
        arr = data[start:end].astype(np.float32)
        ts = timestamps[start:end].astype(np.float32)
        np.save(os.path.join(output_dir, f'{name}_data.npy'), arr)
        np.save(os.path.join(output_dir, f'{name}_timestamps.npy'), ts)
        print(f'{name}_data shape: {arr.shape} ; {name}_timestamps shape: {ts.shape}')
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
