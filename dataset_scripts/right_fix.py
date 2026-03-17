import numpy as np

def check_npy(path):
    arr = np.load(path)
    print(f"{path}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  NaNs: {np.isnan(arr).sum()}, Infs: {np.isinf(arr).sum()}")
    print(f"  min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")

check_npy('dataset/right/data/data_time_domain.npy')
check_npy('dataset/left/data/data_time_domain.npy')
check_npy('dataset/chest/data/data_time_domain.npy')

data = np.load('dataset/right/data/data_time_domain.npy')  # shape: (5760, 460, 8)
print(f"Shape: {data.shape}")

# Per-channel stats
for ch in range(data.shape[2]):
    ch_data = data[:, :, ch]
    print(f"Channel {ch}: min={ch_data.min()}, max={ch_data.max()}, mean={ch_data.mean()}, std={ch_data.std()}")

# Optionally, flag samples with extreme values
threshold = 10000
extreme_indices = np.argwhere(np.abs(data) > threshold)
if extreme_indices.size > 0:
    print(f"\nSamples with |value| > {threshold}:")
    for idx in extreme_indices:
        print(f"Sample {idx[0]}, Time {idx[1]}, Channel {idx[2]}: value={data[tuple(idx)]}")
else:
    print(f"\nNo values exceed |{threshold}|.")

# data_path = 'dataset/right/data/data_time_domain.npy'
# data = np.load(data_path)

# s, t, c0, c2 = 5208, 247, 0, 2

# # Interpolate using the mean of the previous and next time steps for each channel
# for ch in [c0, c2]:
#     prev_val = data[s, t-1, ch]
#     next_val = data[s, t+1, ch]
#     data[s, t, ch] = (prev_val + next_val) / 2

# # Save the cleaned data (overwrite or use a new file)
# np.save(data_path, data)
# print("Outlier values interpolated and data saved.")