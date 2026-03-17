

import numpy as np
import os

def resample_channel(ts, vals, target_n):
	t_uniform = np.linspace(ts[0], ts[-1], target_n)
	return np.interp(t_uniform, ts, vals)

# Parameters
input_data_path = os.path.join('dataset', 'chest', 'data', 'data_time_domain.npy')
input_label_path = os.path.join('dataset', 'chest', 'labels', 'labels.npy')
input_groups_path = os.path.join('dataset', 'chest', 'labels', 'groups.npy')

output_data_dir = os.path.join('dataset', 'chest_downsampled', 'data')
output_label_dir = os.path.join('dataset', 'chest_downsampled', 'labels')
os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

output_data_path = os.path.join(output_data_dir, 'data_time_domain.npy')
output_label_path = os.path.join(output_label_dir, 'labels.npy')
output_groups_path = os.path.join(output_label_dir, 'groups.npy')

target_n = 460  # Target window length

# Load data
X = np.load(input_data_path)  # shape: (n_windows, window_len, n_channels)
labels = np.load(input_label_path)
groups = np.load(input_groups_path)
print(f"Original shape: {X.shape}")
n_windows, window_len, n_channels = X.shape

# Assume uniform timestamps for each window (as in fusion script)
ts_orig = np.linspace(0, 1, window_len)
ts_target = np.linspace(0, 1, target_n)

X_down = np.empty((n_windows, target_n, n_channels), dtype=X.dtype)
for i in range(n_windows):
	for c in range(n_channels):
		X_down[i, :, c] = np.interp(ts_target, ts_orig, X[i, :, c])

print(f"Downsampled shape: {X_down.shape}")
np.save(output_data_path, X_down)
np.save(output_label_path, labels)
np.save(output_groups_path, groups)
print(f"Saved downsampled data to: {output_data_path}")
print(f"Saved labels to: {output_label_path}")
print(f"Saved groups to: {output_groups_path}")
