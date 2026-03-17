"""
Data leakage inspection for the fall-detection dataset.

Checks performed
----------------
1.  Shape & basic integrity  (shapes match, no NaN/Inf)
2.  Group inventory          (unique groups, samples per group, label mix per group)
3.  Label balance            (global and per-group)
4.  Exact-duplicate windows  (identical rows that cross group boundaries)
5.  Near-duplicate windows   (cosine similarity > 0.9999 across groups)
6.  Normalisation / scaling  (any pre-normalised channel? mean≈0 & std≈1 globally?)
7.  LOGO split simulation    (verify no subject appears in both train and test per fold)
8.  Within-group duplicate   (identical rows within the same group – benign but flagged)
"""

import numpy as np
from itertools import combinations
from collections import Counter

DATASET_DIR = r"d:\TCC\fall-detect\dataset\chest_left"

# ── Load ─────────────────────────────────────────────────────────────────────
X  = np.load(rf"{DATASET_DIR}\data\data_time_domain.npy")   # (N, T, C)
y  = np.load(rf"{DATASET_DIR}\labels\labels.npy")
gr = np.load(rf"{DATASET_DIR}\labels\groups.npy")

SEP = "=" * 70

# ── 1. Shape & integrity ──────────────────────────────────────────────────────
print(SEP)
print("1. SHAPE & INTEGRITY")
print(SEP)
print(f"   X shape  : {X.shape}   dtype={X.dtype}")
print(f"   y shape  : {y.shape}   dtype={y.dtype}")
print(f"   gr shape : {gr.shape}  dtype={gr.dtype}")

shape_ok = X.shape[0] == y.shape[0] == gr.shape[0]
print(f"   Shapes aligned  : {'OK' if shape_ok else 'MISMATCH!'}")
print(f"   NaN in X        : {np.isnan(X).sum()}")
print(f"   Inf in X        : {np.isinf(X).sum()}")
print(f"   NaN in y        : {np.isnan(y.astype(float)).sum()}")
print(f"   X value range   : [{X.min():.4f}, {X.max():.4f}]")

# ── 2. Group inventory ────────────────────────────────────────────────────────
print()
print(SEP)
print("2. GROUP INVENTORY")
print(SEP)
unique_groups = np.unique(gr)
print(f"   Unique groups ({len(unique_groups)}): {sorted(unique_groups.tolist())}")
print()
print(f"   {'Group':>7} | {'Samples':>8} | {'Label 0':>8} | {'Label 1':>8} | {'% fall':>7}")
print(f"   {'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
for g in sorted(unique_groups.tolist()):
    mask = gr == g
    yg   = y[mask]
    n0, n1 = (yg == 0).sum(), (yg == 1).sum()
    pct  = 100 * n1 / len(yg) if len(yg) else 0
    print(f"   {g:>7} | {len(yg):>8} | {n0:>8} | {n1:>8} | {pct:>6.1f}%")
print()
counts = Counter(gr.tolist())
sizes  = list(counts.values())
print(f"   Min samples/group : {min(sizes)}  Max : {max(sizes)}  Mean : {np.mean(sizes):.1f}")

# ── 3. Global label balance ───────────────────────────────────────────────────
print()
print(SEP)
print("3. GLOBAL LABEL BALANCE")
print(SEP)
n0, n1 = (y == 0).sum(), (y == 1).sum()
print(f"   Label 0 (ADL / non-fall) : {n0}  ({100*n0/len(y):.1f}%)")
print(f"   Label 1 (fall)           : {n1}  ({100*n1/len(y):.1f}%)")
print(f"   Imbalance ratio (0:1)    : {n0/max(n1,1):.2f}")

# ── 4. Exact cross-group duplicates ──────────────────────────────────────────
print()
print(SEP)
print("4. EXACT CROSS-GROUP DUPLICATES (windows from DIFFERENT groups with identical data)")
print(SEP)

# Flatten each sample to a 1-D byte hash for speed
N = X.shape[0]
flat = X.reshape(N, -1)

# Use a dict: bytes(row) -> list of (index, group)
from collections import defaultdict
hash_map = defaultdict(list)
for i in range(N):
    key = flat[i].tobytes()
    hash_map[key].append(i)

cross_group_dup_count = 0
cross_group_dup_examples = []
within_group_dup_count   = 0

for key, idxs in hash_map.items():
    if len(idxs) < 2:
        continue
    gs = [gr[i] for i in idxs]
    if len(set(gs)) == 1:
        within_group_dup_count += 1
    else:
        cross_group_dup_count += 1
        if len(cross_group_dup_examples) < 5:
            cross_group_dup_examples.append((idxs, gs))

if cross_group_dup_count == 0:
    print("   No exact cross-group duplicates found.")
else:
    print(f"   WARNING: {cross_group_dup_count} exact cross-group duplicate window(s) found!")
    for idxs, gs in cross_group_dup_examples:
        labels_dup = [y[i] for i in idxs]
        print(f"     indices={idxs}  groups={gs}  labels={labels_dup}")

print(f"   (Within-group duplicates: {within_group_dup_count} – benign for long-activity windows)")

# ── 5. Near-duplicate cross-group check (cosine similarity) ──────────────────
print()
print(SEP)
print("5. NEAR-DUPLICATE CROSS-GROUP CHECK (cosine sim > 0.9999 across groups)")
print(SEP)

# Normalise each row for cosine similarity
norms  = np.linalg.norm(flat, axis=1, keepdims=True)
norms  = np.where(norms == 0, 1, norms)          # avoid div/0
normed = flat / norms

# To keep this tractable we sample at most 3000 rows
rng = np.random.default_rng(42)
sample_size = min(3000, N)
sample_idx  = rng.choice(N, size=sample_size, replace=False)
sample_normed  = normed[sample_idx].astype(np.float32)
sample_groups  = gr[sample_idx]

# Compute full cosine matrix in chunks to avoid memory blow-up
CHUNK = 500
near_dup_pairs = []
for i in range(0, sample_size, CHUNK):
    chunk_i = sample_normed[i:i+CHUNK]
    sims = chunk_i @ sample_normed.T   # (chunk, sample_size)
    # zero out self and lower triangle via indices
    for ci in range(len(chunk_i)):
        global_i = i + ci
        for j in range(global_i + 1, sample_size):
            if sims[ci, j] > 0.9999:
                gi, gj = sample_groups[global_i], sample_groups[j]
                if gi != gj:
                    near_dup_pairs.append((sample_idx[global_i], sample_idx[j],
                                           gi, gj, float(sims[ci, j])))
                    if len(near_dup_pairs) >= 20:
                        break
        if len(near_dup_pairs) >= 20:
            break
    if len(near_dup_pairs) >= 20:
        break

if not near_dup_pairs:
    print(f"   No near-duplicate cross-group pairs found (sample={sample_size}).")
else:
    print(f"   WARNING: {len(near_dup_pairs)} near-duplicate cross-group pair(s) found!")
    for idx_i, idx_j, gi, gj, sim in near_dup_pairs[:5]:
        print(f"     idx {idx_i} (group={gi}, y={y[idx_i]}) <-> "
              f"idx {idx_j} (group={gj}, y={y[idx_j]})  sim={sim:.6f}")

# ── 6. Normalisation leakage check ───────────────────────────────────────────
print()
print(SEP)
print("6. NORMALISATION/SCALING LEAKAGE CHECK")
print(SEP)
print("   (If data is already standardised to ~N(0,1) globally it may have been")
print("    scaled BEFORE the train/test split, which is a leakage vector.)")
print()
for c in range(X.shape[2]):
    ch = X[:, :, c].ravel()
    print(f"   Channel {c:2d}: mean={ch.mean():>10.4f}  std={ch.std():>9.4f}  "
          f"min={ch.min():>10.4f}  max={ch.max():>10.4f}")

global_mean  = X.ravel().mean()
global_std   = X.ravel().std()
looks_scaled = abs(global_mean) < 0.1 and 0.8 < global_std < 1.2
print()
if looks_scaled:
    print("   WARNING: data appears to be globally standardised (mean≈0, std≈1).")
    print("   Verify that scaling was done INSIDE each LOGO fold (not before splitting).")
else:
    print("   OK – data does NOT appear to be pre-standardised globally.")

# ── 7. LOGO split sanity ─────────────────────────────────────────────────────
print()
print(SEP)
print("7. LOGO SPLIT SANITY (simulated)")
print(SEP)
leakage_found = False
for g in sorted(unique_groups.tolist()):
    test_mask  = gr == g
    train_mask = ~test_mask
    test_groups  = set(np.unique(gr[test_mask]).tolist())
    train_groups = set(np.unique(gr[train_mask]).tolist())
    overlap = test_groups & train_groups
    if overlap:
        print(f"   LEAKAGE: group(s) {overlap} appear in BOTH train and test for fold g={g}!")
        leakage_found = True
if not leakage_found:
    print("   OK – every LOGO fold has a clean train/test group separation.")

# ── 8. Per-channel feature variance across groups ────────────────────────────
print()
print(SEP)
print("8. PER-CHANNEL MEAN VARIANCE ACROSS GROUPS (flag if any group is an outlier)")
print(SEP)
chan_names = ["mag_acc", "acc_x", "acc_y", "acc_z", "mag_gyr", "gyr_x", "gyr_y", "gyr_z",
              "mag_acc2", "acc_x2", "acc_y2", "acc_z2", "mag_gyr2", "gyr_x2", "gyr_y2", "gyr_z2"]
chan_names = chan_names[:X.shape[2]]

group_channel_means = {}
for g in sorted(unique_groups.tolist()):
    group_channel_means[g] = X[gr == g].mean(axis=(0, 1))  # (C,)

all_means = np.stack(list(group_channel_means.values()))   # (G, C)
overall_mean = all_means.mean(axis=0)
overall_std  = all_means.std(axis=0)

print(f"   {'Group':>7}", end="")
for c in range(X.shape[2]):
    print(f"  {chan_names[c]:>10}", end="")
print()
outlier_groups = []
for g in sorted(unique_groups.tolist()):
    means = group_channel_means[g]
    z_scores = np.abs((means - overall_mean) / np.where(overall_std == 0, 1, overall_std))
    is_outlier = z_scores.max() > 3.0
    if is_outlier:
        outlier_groups.append(g)
    marker = " <-- OUTLIER (z>3)" if is_outlier else ""
    print(f"   {g:>7}", end="")
    for m in means:
        print(f"  {m:>10.3f}", end="")
    print(marker)

if not outlier_groups:
    print("\n   No statistical outlier groups detected.")
else:
    print(f"\n   WARNING: groups {outlier_groups} have channel means >3 std from the group average.")
    print("   This could indicate subject-specific biases that inflate LOGO performance.")

print()
print(SEP)
print("INSPECTION COMPLETE")
print(SEP)

# ── 9. FUSED WINDOW ALIGNMENT CHECK ─────────────────────────────────────────
print()
print(SEP)
print("9. FUSED WINDOW ALIGNMENT CHECK (multi-sensor consistency)")
print(SEP)

# Only run this check if the number of channels suggests a fused dataset (e.g., >8 channels)
if X.shape[2] > 8 and X.shape[2] % 8 == 0:
    n_pos = X.shape[2] // 8
    print(f"   Detected {n_pos} fused sensor positions (channels: {X.shape[2]})")
    # For a few random samples, print per-position stats
    rng = np.random.default_rng(123)
    sample_idx = rng.choice(X.shape[0], size=min(5, X.shape[0]), replace=False)
    for idx in sample_idx:
        print(f"\n   Sample idx {idx}  (group={gr[idx]}, label={y[idx]})")
        for pos in range(n_pos):
            ch_start = pos * 8
            ch_end = (pos + 1) * 8
            window = X[idx, :, ch_start:ch_end]
            # Print min/max timestamp if available, else just stats
            print(f"     Position {pos+1}: mean={window.mean():.3f} std={window.std():.3f} min={window.min():.3f} max={window.max():.3f}")
    # Check that all windows for a group have the same count
    group_counts = {}
    for g in np.unique(gr):
        group_counts[g] = (gr == g).sum()
    print("\n   Fused window count per group:")
    for g in sorted(group_counts):
        print(f"     Group {g}: {group_counts[g]} windows")
    # Optionally, check for identical windows across positions (should not happen)
    identical_cross_pos = 0
    for idx in range(X.shape[0]):
        base = X[idx, :, 0:8]
        for pos in range(1, n_pos):
            comp = X[idx, :, pos*8:(pos+1)*8]
            if np.allclose(base, comp):
                identical_cross_pos += 1
    if identical_cross_pos > 0:
        print(f"\n   WARNING: {identical_cross_pos} fused windows have identical data across positions (possible error in fusion)")
    else:
        print("\n   No fused windows are exactly identical across positions (OK)")
else:
    print("   (Not a fused dataset or only one position; skipping alignment check.)")

print()
