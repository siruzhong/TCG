import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0, rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.edgecolor": ".15",
    "grid.linestyle": "--",
    "axes.linewidth": 1.2,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 10,
})

horizons = [24, 36, 48, 60]

orth_data = {
    'H=24': [3.456, 3.380, 3.458, 3.460],
    'H=36': [4.008, 3.923, 3.925, 4.016],
    'H=48': [2.887, 2.887, 2.886, 2.888],
    'H=60': [2.726, 2.726, 2.726, 2.727],
}
orth_lambdas = [1e-5, 0.0001, 0.001, 0.1]
orth_avg = [3.269, 3.229, 3.249, 3.273]

k_data = {
    'H=24': [3.356, 3.380, 3.542, 3.571],
    'H=36': [4.115, 3.923, 3.954, 3.611],
    'H=48': [2.846, 2.887, 2.931, 2.952],
    'H=60': [2.801, 2.726, 2.746, 2.809],
}
k_values = [4, 8, 16, 32]
k_avg = [3.279, 3.229, 3.293, 3.236]

conv_labels = ['1', '3', '5', '7', '3+7']
conv_x = list(range(len(conv_labels)))

patch_conv_data = {
    'H=24': [3.108, 3.266, 3.582, 3.460, 3.464],
    'H=36': [3.892, 4.053, 4.138, 3.969, 3.929],
    'H=48': [2.887, 2.974, 2.873, 2.857, 2.938],
    'H=60': [2.676, 2.646, 2.801, 2.711, 2.678],
}
patch_conv_avg = [3.141, 3.235, 3.348, 3.249, 3.252]

cross_conv_data = {
    'H=24': [4.954, 5.226, 4.905, 4.666, 4.591],
    'H=36': [5.003, 5.158, 5.523, 5.059, 6.088],
    'H=48': [5.124, 5.470, 4.792, 5.737, 5.522],
    'H=60': [5.195, 4.955, 5.341, 5.162, 4.942],
}
cross_conv_avg = [5.069, 5.202, 5.141, 5.156, 5.286]

fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.2), constrained_layout=True)

colors = ['#2171B5', '#6A51A3', '#F4A460', '#D94801']
markers = ['o', 's', '^', 'D']

for i, h in enumerate(horizons):
    axes[0].plot(orth_lambdas, orth_data[f'H={h}'], marker=markers[i], markersize=7,
                 linewidth=2.0, color=colors[i], label=f'H={h}', zorder=3)

axes[0].plot(orth_lambdas, orth_avg, color='gray', linewidth=1.5, linestyle='--',
             label='Avg', zorder=2, alpha=0.5)

for i, h in enumerate(horizons):
    axes[0].scatter([0.0001], [orth_data[f'H={h}'][1]], marker='*', s=150,
                     color=colors[i], edgecolors='black', linewidth=0.8, zorder=5)

axes[0].set_xscale('log')
axes[0].set_xticks([1e-5, 0.0001, 0.001, 0.1])
axes[0].set_xticklabels(['0', '0.0001', '0.001', '0.1'])
axes[0].set_title(r'(a) Orthogonal Regularization $\lambda$', fontweight='bold', fontsize=11)
axes[0].set_xlabel(r'Orthogonal Regularization $\lambda$')
axes[0].set_ylabel('MSE')
axes[0].legend(title='Horizon', loc='upper right', framealpha=0.9, edgecolor='gray')
axes[0].set_ylim([2.65, 4.15])
axes[0].grid(True, linestyle='--', alpha=0.25)

for i, h in enumerate(horizons):
    axes[1].plot(k_values, k_data[f'H={h}'], marker=markers[i], markersize=7,
                 linewidth=2.0, color=colors[i], label=f'H={h}', zorder=3)

axes[1].plot(k_values, k_avg, color='gray', linewidth=1.5, linestyle='--',
             label='Avg', zorder=2, alpha=0.5)

for i, h in enumerate(horizons):
    axes[1].scatter([8], [k_data[f'H={h}'][1]], marker='*', s=150,
                    color=colors[i], edgecolors='black', linewidth=0.8, zorder=5)

axes[1].set_xticks([4, 8, 16, 32])
axes[1].set_xticklabels(['4', '8', '16', '32'])
axes[1].set_title(r'(b) Number of Patterns K', fontweight='bold', fontsize=11)
axes[1].set_xlabel(r'Number of Patterns K')
axes[1].set_ylabel('MSE')
axes[1].legend(title='Horizon', loc='upper right', framealpha=0.9, edgecolor='gray')
axes[1].set_ylim([2.65, 4.15])
axes[1].grid(True, linestyle='--', alpha=0.25)

for i, h in enumerate(horizons):
    axes[2].plot(conv_x, patch_conv_data[f'H={h}'], marker=markers[i], markersize=7,
                 linewidth=2.0, color=colors[i], label=f'H={h}', zorder=3)

axes[2].plot(conv_x, patch_conv_avg, color='gray', linewidth=1.5, linestyle='--',
             label='Avg', zorder=2, alpha=0.5)

# Highlight 24->24 preference for PatchTST at k=1.
axes[2].scatter([0], [patch_conv_data['H=24'][0]], marker='*', s=150,
                color=colors[0], edgecolors='black', linewidth=0.8, zorder=5)

axes[2].set_xticks(conv_x)
axes[2].set_xticklabels(conv_labels)
axes[2].set_title(r'(c) Conv Kernels (PatchTST)', fontweight='bold', fontsize=11)
axes[2].set_xlabel(r'Conv Kernels $k$')
axes[2].set_ylabel('MSE')
axes[2].legend(title='Horizon', loc='upper right', framealpha=0.9, edgecolor='gray')
axes[2].set_ylim([2.60, 4.20])
axes[2].grid(True, linestyle='--', alpha=0.25)

for i, h in enumerate(horizons):
    axes[3].plot(conv_x, cross_conv_data[f'H={h}'], marker=markers[i], markersize=7,
                 linewidth=2.0, color=colors[i], label=f'H={h}', zorder=3)

axes[3].plot(conv_x, cross_conv_avg, color='gray', linewidth=1.5, linestyle='--',
             label='Avg', zorder=2, alpha=0.5)

# Highlight 24->24 preference for Crossformer at k=(3,7).
axes[3].scatter([4], [cross_conv_data['H=24'][4]], marker='*', s=150,
                color=colors[0], edgecolors='black', linewidth=0.8, zorder=5)

axes[3].set_xticks(conv_x)
axes[3].set_xticklabels(conv_labels)
axes[3].set_title(r'(d) Conv Kernels (Crossformer)', fontweight='bold', fontsize=11)
axes[3].set_xlabel(r'Conv Kernels $k$')
axes[3].set_ylabel('MSE')
axes[3].legend(title='Horizon', loc='upper right', framealpha=0.9, edgecolor='gray')
axes[3].set_ylim([4.40, 6.20])
axes[3].grid(True, linestyle='--', alpha=0.25)

plt.savefig('parameter_sensitivity_analysis.pdf', bbox_inches='tight', dpi=300)
plt.show()
