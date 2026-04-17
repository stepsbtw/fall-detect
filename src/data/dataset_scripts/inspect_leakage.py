"""
Data leakage inspection for one or more fall-detection datasets.

Checks performed
----------------
1.  Shape & basic integrity
2.  Group inventory
3.  Label balance
4.  Exact-duplicate windows across groups
5.  Near-duplicate windows across groups
6.  Normalisation / scaling leakage
7.  LOGO split simulation
8.  Per-channel mean variance across groups
9.  Fused window alignment check
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np


SEP = "=" * 70


def load_dataset(dataset_dir):
    data_path = os.path.join(dataset_dir, "data", "data_time_domain.npy")
    labels_path = os.path.join(dataset_dir, "labels", "labels.npy")
    groups_path = os.path.join(dataset_dir, "labels", "groups.npy")

    if not (os.path.isfile(data_path) and os.path.isfile(labels_path) and os.path.isfile(groups_path)):
        raise FileNotFoundError(
            f"Missing required files in {dataset_dir}. "
            f"Expected data/data_time_domain.npy, labels/labels.npy, labels/groups.npy"
        )

    X = np.load(data_path)
    y = np.load(labels_path)
    gr = np.load(groups_path)
    return X, y, gr


def inspect_dataset(dataset_dir, near_dup_threshold=0.9999, near_dup_sample_size=3000):
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    print("\n" + "#" * 90)
    print(f"DATASET: {dataset_name}")
    print("#" * 90)

    X, y, gr = load_dataset(dataset_dir)

    # 1. Shape & integrity
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

    # 2. Group inventory
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
        yg = y[mask]
        n0, n1 = (yg == 0).sum(), (yg == 1).sum()
        pct = 100 * n1 / len(yg) if len(yg) else 0
        print(f"   {g:>7} | {len(yg):>8} | {n0:>8} | {n1:>8} | {pct:>6.1f}%")
    print()
    counts = Counter(gr.tolist())
    sizes = list(counts.values())
    print(f"   Min samples/group : {min(sizes)}  Max : {max(sizes)}  Mean : {np.mean(sizes):.1f}")

    # 3. Global label balance
    print()
    print(SEP)
    print("3. GLOBAL LABEL BALANCE")
    print(SEP)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    print(f"   Label 0 (ADL / non-fall) : {n0}  ({100*n0/len(y):.1f}%)")
    print(f"   Label 1 (fall)           : {n1}  ({100*n1/len(y):.1f}%)")
    print(f"   Imbalance ratio (0:1)    : {n0/max(n1,1):.2f}")

    # 4. Exact cross-group duplicates
    print()
    print(SEP)
    print("4. EXACT CROSS-GROUP DUPLICATES")
    print(SEP)

    N = X.shape[0]
    flat = X.reshape(N, -1)

    hash_map = defaultdict(list)
    for i in range(N):
        key = flat[i].tobytes()
        hash_map[key].append(i)

    cross_group_dup_count = 0
    cross_group_dup_examples = []
    within_group_dup_count = 0

    for _, idxs in hash_map.items():
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

    print(f"   (Within-group duplicates: {within_group_dup_count})")

    # 5. Near-duplicate cross-group check
    print()
    print(SEP)
    print(f"5. NEAR-DUPLICATE CROSS-GROUP CHECK (cosine sim > {near_dup_threshold})")
    print(SEP)

    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = flat / norms

    rng = np.random.default_rng(42)
    sample_size = min(near_dup_sample_size, N)
    sample_idx = rng.choice(N, size=sample_size, replace=False)
    sample_normed = normed[sample_idx].astype(np.float32)
    sample_groups = gr[sample_idx]

    CHUNK = 500
    near_dup_pairs = []
    for i in range(0, sample_size, CHUNK):
        chunk_i = sample_normed[i:i + CHUNK]
        sims = chunk_i @ sample_normed.T
        for ci in range(len(chunk_i)):
            global_i = i + ci
            for j in range(global_i + 1, sample_size):
                if sims[ci, j] > near_dup_threshold:
                    gi, gj = sample_groups[global_i], sample_groups[j]
                    if gi != gj:
                        near_dup_pairs.append(
                            (sample_idx[global_i], sample_idx[j], gi, gj, float(sims[ci, j]))
                        )
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
            print(
                f"     idx {idx_i} (group={gi}, y={y[idx_i]}) <-> "
                f"idx {idx_j} (group={gj}, y={y[idx_j]})  sim={sim:.6f}"
            )

    # 6. Normalisation leakage check
    print()
    print(SEP)
    print("6. NORMALISATION/SCALING LEAKAGE CHECK")
    print(SEP)
    for c in range(X.shape[2]):
        ch = X[:, :, c].ravel()
        print(
            f"   Channel {c:2d}: mean={ch.mean():>10.4f}  std={ch.std():>9.4f}  "
            f"min={ch.min():>10.4f}  max={ch.max():>10.4f}"
        )

    global_mean = X.ravel().mean()
    global_std = X.ravel().std()
    looks_scaled = abs(global_mean) < 0.1 and 0.8 < global_std < 1.2

    print()
    if looks_scaled:
        print("   WARNING: data appears to be globally standardised (mean≈0, std≈1).")
        print("   Verify scaling was done inside each LOGO fold.")
    else:
        print("   OK – data does NOT appear to be pre-standardised globally.")

    # 7. LOGO split sanity
    print()
    print(SEP)
    print("7. LOGO SPLIT SANITY")
    print(SEP)
    leakage_found = False
    for g in sorted(unique_groups.tolist()):
        test_mask = gr == g
        train_mask = ~test_mask
        test_groups = set(np.unique(gr[test_mask]).tolist())
        train_groups = set(np.unique(gr[train_mask]).tolist())
        overlap = test_groups & train_groups
        if overlap:
            print(f"   LEAKAGE: group(s) {overlap} appear in BOTH train and test for fold g={g}!")
            leakage_found = True
    if not leakage_found:
        print("   OK – every LOGO fold has a clean train/test group separation.")

    # 8. Per-channel mean variance across groups
    print()
    print(SEP)
    print("8. PER-CHANNEL MEAN VARIANCE ACROSS GROUPS")
    print(SEP)

    chan_names = [
        "mag_acc", "acc_x", "acc_y", "acc_z", "mag_gyr", "gyr_x", "gyr_y", "gyr_z",
        "mag_acc2", "acc_x2", "acc_y2", "acc_z2", "mag_gyr2", "gyr_x2", "gyr_y2", "gyr_z2",
        "mag_acc3", "acc_x3", "acc_y3", "acc_z3", "mag_gyr3", "gyr_x3", "gyr_y3", "gyr_z3",
    ]
    chan_names = chan_names[:X.shape[2]]

    group_channel_means = {}
    for g in sorted(unique_groups.tolist()):
        group_channel_means[g] = X[gr == g].mean(axis=(0, 1))

    all_means = np.stack(list(group_channel_means.values()))
    overall_mean = all_means.mean(axis=0)
    overall_std = all_means.std(axis=0)

    print(f"   {'Group':>7}", end="")
    for c in range(X.shape[2]):
        name = chan_names[c] if c < len(chan_names) else f"ch{c}"
        print(f"  {name:>10}", end="")
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

    # 9. Fused window alignment check
    print()
    print(SEP)
    print("9. FUSED WINDOW ALIGNMENT CHECK")
    print(SEP)

    if X.shape[2] > 8 and X.shape[2] % 8 == 0:
        n_pos = X.shape[2] // 8
        print(f"   Detected {n_pos} fused sensor positions (channels: {X.shape[2]})")

        rng = np.random.default_rng(123)
        inspect_idx = rng.choice(X.shape[0], size=min(5, X.shape[0]), replace=False)
        for idx in inspect_idx:
            print(f"\n   Sample idx {idx}  (group={gr[idx]}, label={y[idx]})")
            for pos in range(n_pos):
                ch_start = pos * 8
                ch_end = (pos + 1) * 8
                window = X[idx, :, ch_start:ch_end]
                print(
                    f"     Position {pos+1}: mean={window.mean():.3f} "
                    f"std={window.std():.3f} min={window.min():.3f} max={window.max():.3f}"
                )

        identical_cross_pos = 0
        for idx in range(X.shape[0]):
            base = X[idx, :, 0:8]
            for pos in range(1, n_pos):
                comp = X[idx, :, pos * 8:(pos + 1) * 8]
                if np.allclose(base, comp):
                    identical_cross_pos += 1

        if identical_cross_pos > 0:
            print(f"\n   WARNING: {identical_cross_pos} fused windows have identical data across positions.")
        else:
            print("\n   No fused windows are exactly identical across positions (OK)")
    else:
        print("   (Not a fused dataset or only one position; skipping alignment check.)")

    print()
    return {
        "dataset": dataset_name,
        "N": int(X.shape[0]),
        "channels": int(X.shape[2]),
        "exact_cross_group_duplicates": int(cross_group_dup_count),
        "near_cross_group_duplicates_found": int(len(near_dup_pairs)),
        "looks_globally_scaled": bool(looks_scaled),
        "outlier_groups": outlier_groups,
    }


def discover_dataset_dirs(dataset_root):
    dataset_dirs = []
    for name in sorted(os.listdir(dataset_root)):
        path = os.path.join(dataset_root, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "data", "data_time_domain.npy")) and \
           os.path.isfile(os.path.join(path, "labels", "labels.npy")) and \
           os.path.isfile(os.path.join(path, "labels", "groups.npy")):
            dataset_dirs.append(path)
    return dataset_dirs


def main():
    parser = argparse.ArgumentParser(description="Inspect leakage for one or more datasets")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"d:\TCC\fall-detect\dataset",
        help="Root folder containing dataset subfolders",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset folder names to inspect. If omitted, inspect all valid subfolders.",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)

    if args.datasets:
        dataset_dirs = [os.path.join(dataset_root, name) for name in args.datasets]
    else:
        dataset_dirs = discover_dataset_dirs(dataset_root)

    if not dataset_dirs:
        raise FileNotFoundError(f"No valid dataset folders found under: {dataset_root}")

    summaries = []
    for dataset_dir in dataset_dirs:
        summaries.append(inspect_dataset(dataset_dir))

    print("\n" + "#" * 90)
    print("SUMMARY")
    print("#" * 90)
    print(
        f"{'dataset':<28} {'N':>8} {'C':>4} {'exact_dup':>10} "
        f"{'near_dup':>9} {'scaled?':>8} {'outlier_groups':>20}"
    )
    for s in summaries:
        print(
            f"{s['dataset']:<28} {s['N']:>8} {s['channels']:>4} "
            f"{s['exact_cross_group_duplicates']:>10} "
            f"{s['near_cross_group_duplicates_found']:>9} "
            f"{str(s['looks_globally_scaled']):>8} "
            f"{str(s['outlier_groups']):>20}"
        )


if __name__ == "__main__":
    main()