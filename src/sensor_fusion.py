import numpy as np

CANONICAL_SENSORS = ("chest", "left", "right")
BLOCK_SIZE = 8


def sensors_from_scenario(scenario: str):
    base = scenario
    if base.endswith("_NW"):
        base = base[:-3]
    if "_IVG" in base:
        base = base.split("_IVG", 1)[0]
    for suffix in ("_SC", "_NM", "_OM"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if base.endswith("_T") or base.endswith("_F"):
        base = base[:-2]
    parts = [p for p in base.split("_") if p and p not in {"cross", "sensor"}]
    sensors = [p for p in parts if p in CANONICAL_SENSORS]
    if not sensors:
        raise ValueError(f"Could not infer sensors from scenario '{scenario}'")
    return sensors


def infer_sensors_from_width(num_channels: int):
    n_blocks = int(num_channels) // BLOCK_SIZE
    return list(CANONICAL_SENSORS[:n_blocks])


def sensor_block_map(sensors):
    return {
        sensor: (idx * BLOCK_SIZE, (idx + 1) * BLOCK_SIZE)
        for idx, sensor in enumerate(sensors)
    }


def sensor_block_map_for_scenario(scenario: str):
    return sensor_block_map(sensors_from_scenario(scenario))


def zero_sensor_blocks(X, sensors_to_zero, scenario: str = None, sensors=None, inplace: bool = False):
    if sensors is None:
        if scenario is None:
            sensors = infer_sensors_from_width(X.shape[-1])
        else:
            sensors = sensors_from_scenario(scenario)
    sensors_to_zero = list(sensors_to_zero)
    out = X if inplace else np.array(X, copy=True)
    block_map = sensor_block_map(sensors)
    for sensor in sensors_to_zero:
        if sensor not in block_map:
            continue
        start, end = block_map[sensor]
        out[..., start:end] = 0.0
    return out


def sample_sensor_dropout_mask(rng, available_sensors, p=0.5, max_off=1, allow_no_dropout=True):
    available_sensors = list(available_sensors)
    if len(available_sensors) == 0:
        return []
    if rng.random() >= p:
        return []
    upper = min(max(int(max_off), 1), len(available_sensors))
    lower = 0 if allow_no_dropout else 1
    n_drop = int(rng.integers(lower if lower > 0 else 1, upper + 1))
    if n_drop <= 0:
        return []
    return list(rng.choice(available_sensors, size=n_drop, replace=False))


def apply_sensor_dropout_batch(
    X,
    scenario: str = None,
    sensors=None,
    p=0.5,
    max_off=1,
    allow_no_dropout=True,
    seed=None,
):
    sensors = list(sensors or (sensors_from_scenario(scenario) if scenario else infer_sensors_from_width(X.shape[-1])))
    rng = np.random.default_rng(seed)
    out = np.array(X, copy=True)
    applied = []
    for idx in range(len(out)):
        dropped = sample_sensor_dropout_mask(
            rng,
            sensors,
            p=p,
            max_off=max_off,
            allow_no_dropout=allow_no_dropout,
        )
        if dropped:
            out[idx] = zero_sensor_blocks(out[idx], dropped, sensors=sensors, inplace=False)
        applied.append(dropped)
    return out, applied


def expand_to_canonical(X, source_scenario: str, target_sensors=CANONICAL_SENSORS, fill_value=0.0):
    source_sensors = sensors_from_scenario(source_scenario)
    source_map = sensor_block_map(source_sensors)
    target_map = sensor_block_map(target_sensors)
    n, t, _ = X.shape
    out = np.full((n, t, len(target_sensors) * BLOCK_SIZE), fill_value, dtype=X.dtype)
    for sensor in source_sensors:
        s0, s1 = source_map[sensor]
        t0, t1 = target_map[sensor]
        out[:, :, t0:t1] = X[:, :, s0:s1]
    return out


def availability_vector(active_sensors, target_sensors=CANONICAL_SENSORS):
    active = set(active_sensors)
    return np.array([1 if sensor in active else 0 for sensor in target_sensors], dtype=np.float32)
