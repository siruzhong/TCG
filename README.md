# DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting

## Overview

DropoutTS is a sample-adaptive dropout method designed for time series forecasting. Unlike traditional dropout that applies a fixed rate across all samples, DropoutTS dynamically adjusts dropout rates based on the noise level of each sample, estimated through frequency-domain analysis. This approach provides adaptive regularization that is more effective for noisy time series data.

### Key Features

- **Sample-Adaptive Dropout**: Automatically adjusts dropout rates per sample based on noise estimation
- **Frequency-Domain Analysis**: Uses Spectral Flatness Measure (SFM) as a physical anchor for noise estimation
- **Task-Loss Optimization**: Dropout parameters are optimized end-to-end with the forecasting task
- **Easy Integration**: Seamlessly integrates with existing time series forecasting models via callback mechanism
- **Comprehensive Evaluation**: Tested on multiple datasets including synthetic and real-world time series

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 1.10.0
- NumPy >= 1.24.4
- Other dependencies listed in `requirements.txt`

### Install from Source

```bash
# Clone the repository (replace with actual repository URL)
cd DropoutTS
pip install -e .
```

## Quick Start

### Basic Usage

DropoutTS can be easily integrated into your training pipeline using the `DropoutTSCallback`:

```python
from basicts.runners.callback import DropoutTSCallback
from basicts import BasicTSLauncher

# Initialize callback with desired hyperparameters
dropout_callback = DropoutTSCallback(
    p_min=0.1,      # Minimum dropout rate for clean samples
    p_max=0.5       # Maximum dropout rate for noisy samples
)

# Add to your callbacks list
callbacks = [dropout_callback, ...]

# Use with the launcher
launcher = BasicTSLauncher(...)
launcher.train(callbacks=callbacks)
```

### Configuration File

You can also configure DropoutTS through a YAML configuration file:

```yaml
callbacks:
  - name: DropoutTSCallback
    p_min: 0.1
    p_max: 0.5
    init_alpha: 10.0
    init_sensitivity: 5.0
    detrend_method: robust_ols
    use_instance_norm: true
    use_sfm_anchor: true
```

## Method

### Architecture

DropoutTS consists of three main components:

1. **Noise Scorer**: Estimates noise levels using frequency-domain analysis
   - Computes Spectral Flatness Measure (SFM) as a physical anchor
   - Uses Robust Rank Filter (RRF) for noise score computation
   - Supports detrending and instance normalization

2. **Adaptive Rate Computation**: Maps noise scores to dropout rates
   - Uses sigmoid-based mapping with learnable sensitivity parameter
   - Constrains dropout rates between `p_min` and `p_max`
   - Provides per-sample dropout rates

3. **Sample-Adaptive Dropout Layer**: Applies dropout with sample-specific rates
   - Supports per-sample dropout rates
   - Seamless integration with existing models

### Key Innovation

The core innovation of DropoutTS is using frequency-domain analysis to estimate noise levels, which provides a principled way to adapt regularization strength. Samples with higher noise (flatter spectrum) receive higher dropout rates, while cleaner samples (more structured spectrum) receive lower dropout rates.

## Experiments

### Dataset Preparation

#### Synthetic Datasets

The SyntheticTS datasets can be generated using the provided script:

```bash
# Generate all synthetic datasets with different noise levels (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
python scripts/data_preparation/SyntheticTS/generate_training_data.py --generate_all

# Or generate a single dataset with specific noise level
python scripts/data_preparation/SyntheticTS/generate_training_data.py --noise_level 0.3 --num_chunks 100 --chunk_size 336 --num_features 1 --output_suffix "_noise0.3"
```

The synthetic datasets are generated with:
- **Signal Components**: Trend + Quasi-Periodic (with drift) + Transient (Chirp + AM)
- **Noise Types**: Gaussian noise + Heavy-tail noise + Random dropouts
- **Noise Levels**: 0.0 (clean), 0.1, 0.3, 0.5, 0.7, 0.9

#### Real-World Datasets

For real-world datasets (ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather, Illness), please refer to the `datasets/README.md` for download and preparation instructions.

### Supported Datasets

DropoutTS has been evaluated on:

- **Synthetic Datasets**: SyntheticTS with varying noise levels (0.1 to 0.9)
- **Real-World Datasets**: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather, Illness

### Supported Models

DropoutTS is model-agnostic and can be applied to various architectures:

- PatchTST
- Crossformer
- Informer
- iTransformer
- TimeMixer
- TimesNet
- DLinear
- And more...

### Running Experiments

```bash
# Run experiments with DropoutTS
python run_baselines.py --enable_dropout_ts
```

## Results

DropoutTS consistently improves forecasting performance, especially on noisy datasets. Key findings:

- **Noise-Adaptive Regularization**: More effective than fixed dropout rates
- **Improved Generalization**: Better performance on test sets with varying noise levels
- **Model-Agnostic**: Benefits various architectures consistently

For detailed results, please refer to the paper.

## Code Structure

```
DropoutTS/
├── src/basicts/
│   ├── modules/
│   │   └── dropout_ts.py          # Core DropoutTS implementation
│   └── runners/
│       └── callback/
│           └── dynamic_dropout.py  # DropoutTSCallback
├── run_baselines.py                # Main entry for experiments
├── vis/                            # Visualization scripts
└── scripts/                        # Data preparation and table generation
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

This work builds upon an open-source time series framework. We thank the open-source community for their valuable contributions.
