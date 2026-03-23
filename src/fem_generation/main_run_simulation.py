"""
main_run_simulation.py

Generates train / val / test parameter sets from a YAML config file,
then runs beam FEA simulations by calling fea_simulation()

The idea is that
1) validation has values in-distribution (i.e. in the range train), but different from train
2) test has out-of-distribution values (outside the range of train), and I generate it in a way such that there
are test samples that have only one feature (between force_x, force_y, force_z, lengths, width) out of distribution,
and then some that have 2, some 3, and so on. Every test sample has at least one feature out of distribution.
"""

import json
from itertools import combinations, product
from math import ceil
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.fem_generation.beam_fea_solver import fea_simulation

# Round-based equality used when deciding whether two float values should be treated as the same grid value.
DECIMALS = 10
# Small tolerance for interval checks.
TOL = 1e-9
# Threshold used to decide whether a range is effectively constant.
ARE_EQUAL_THRESHOLD = 1e-12


def _train_count(cfg):
    """
    Count how many unique train combinations exist, without materializing the full list.

    Args:
        cfg: dict
            yaml config
        return: int
            number of unique train combinations
    """
    fx, fy, fz, lengths, widths = _train_axis_values(cfg)
    return len(fx) * len(fy) * len(fz) * len(lengths) * len(widths)


def _graph_filepath(params, dir_path):
    """
    Reconstruct the .pt path that fea_simulation will write.

    Args:
        params: dict
            simulation parameter dict
        dir_path: pathlib.Path
            destination directory
        return: pathlib.Path
            graph filepath
    """
    # Keep the same filename convention used before.
    p = params

    # Build the filename deterministically from the simulation parameters.
    name = (f"graphsL{p['L']}_W{p['W']}_D{p['D']}"
            f"_NL{p['NL']}_NW{p['NW']}_ND{p['ND']}"
            f"_E{p['E']}_nu{p['nu']}_rho{p['rho']}"
            f"_em{p['eta_m']}_ek{p['eta_k']}"
            f"_Pix{p['fx']}_Piy{p['fy']}_Piz{p['fz']}"
            f"_T{p['total_time']}"
            f"_Tc{p['cutoff_time_factor'] * p['total_time']}"
            f"_Nsteps{p['num_steps']}")

    # Return the final full filepath.
    return dir_path / f"{name}.pt"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    """
    Load a YAML config file.

    Args:
        path: str
            config filepath
        return: dict
            parsed yaml config
    """
    with open(path) as f:
        return yaml.safe_load(f)


def linspace_list(start, stop, n):
    """
    Return a list from np.linspace with the specified parameters.

    Args:
        start: float
            start of the range
        stop: float
            stop of the range
        n: int
            number of elements
        return: list of float
            list created from np.linspace
    """
    return list(np.linspace(start, stop, n))


def _is_constant_range(start, stop):
    """
    Check whether a numeric range is effectively constant.

    Args:
        start: float
            range start
        stop: float
            range stop
        return: bool
            True if the range is effectively constant
    """
    return abs(float(start) - float(stop)) < ARE_EQUAL_THRESHOLD


def _unique_float_list(values, decimals):
    """
    Delete duplicate float values while preserving order, and return the changed list.

    Args:
        values: list of float
            values to deduplicate
        decimals: int
            number of decimals used to compare values
        return: list of float
            unique values preserving first occurrence order
    """
    unique_values = []
    # Track already seen rounded values.in a set
    seen = set()
    # Visit values in order.
    for value in values:
        # Round before hashing so near-identical float values collapse deterministically.
        rounded_value = round(float(value), decimals)
        # Skip values already seen and mark the seen ones
        if rounded_value in seen:
            continue
        seen.add(rounded_value)
        unique_values.append(float(value))
    return unique_values


def _build_unique_axis_values(start, stop, n, axis_name):
    """
    Build the list of values for one axis, guaranteeing uniqueness by construction.
    Important:
    - If the axis is constant, return a single value only.
      This is the key fix for the train split:
      repeated identical values on an axis should not create repeated simulations.

    Args:
        start: float
            range start
        stop: float
            range stop
        n: int
            requested number of values in config
        axis_name: str
            axis name used only for better error messages
        return: list of float
            unique axis values
    """
    # Guard against invalid configs.
    if int(n) < 1:
        raise ValueError(f"Axis '{axis_name}' must have n >= 1, found n={n}.")
    # If the axis is constant, return only one value.
    # This avoids producing duplicate train combinations and removes the need for combo-level dedup.
    if _is_constant_range(start, stop):
        return [float(start)]
    # Build the raw linearly spaced values.
    raw_values = linspace_list(start, stop, int(n))
    # Deduplicate them using the same float-rounding rule used elsewhere.
    unique_values = _unique_float_list(raw_values, DECIMALS)
    # For a non-constant axis, duplicates would be suspicious.
    # Better to fail loudly than silently change the effective grid size.
    if len(unique_values) != len(raw_values):
        raise ValueError(f"Axis '{axis_name}' lost uniqueness after rounding to {DECIMALS} decimals. "
                         f"Please reduce n or change the numeric range.")

    # Return the unique axis values.
    return unique_values


def _select_evenly_spread(values, n):
    """
    Select n elements from an ordered list so that they are spread as uniformly as possible.

    This is deterministic and non-random.
    It fixes the previous bias where taking vals[:n] always preferred smaller values.

    Args:
        values: list
            ordered candidate values or ordered candidate combos
        n: int
            number of elements to select
        return: list
            deterministically spread subset
    """
    # Guard against invalid requests.
    if int(n) < 0:
        raise ValueError(f"Requested a negative number of values: n={n}.")
    # If the sizes already match, just return a copy.
    if len(values) == int(n):
        return list(values)
    # Total number of available candidates.
    n_candidates = len(values)
    # Build evenly spread integer indices. The 0.5 shift makes the selection use the center of each bin.
    float_indices = (np.arange(int(n)) + 0.5) * n_candidates / int(n)
    indices = np.floor(float_indices).astype(int)
    # Extract the selected values in order.
    selected = [values[int(index)] for index in indices]
    return selected


def vals_excluding(start, stop, n, train_values):
    """
    Generate n values inside [start, stop] that are NOT exact train-grid values.

    The method is:
    1. build a dense ordered linspace
    2. remove the exact train-grid values
    3. pick n values spread uniformly over the remaining ordered candidates

    This keeps determinism, but removes the previous "always take the first values" bias.

    Args:
        start: float
            value starting the range
        stop: float
            value finishing the range
        n: int
            number of values to generate
        train_values: list of float
            exact training-axis values to exclude
        return: list of float
            deterministic, spread, in-range values not matching the train grid
    """
    # Constant axes cannot generate alternative in-range values. Keep one value only so the axis stays unique.
    if _is_constant_range(start, stop):
        return [float(start)]
    # Build the rounded set of train values to exclude.
    train_set = set(np.round(train_values, DECIMALS))
    # Start with a moderate density and increase it only if needed.
    density_factor = 2
    # Keep densifying until there are enough admissible candidates.
    while True:
        # Use a dense count that grows both with the target size and with the train grid size.
        dense_count = density_factor * max(int(n), len(train_values) + 1)

        # Build the dense ordered candidates.
        candidates = linspace_list(start, stop, dense_count)

        # Filter out exact train-grid values.
        filtered_candidates = []
        for candidate in candidates:
            rounded_candidate = round(float(candidate), DECIMALS)
            if rounded_candidate in train_set:
                continue
            filtered_candidates.append(float(candidate))

        # Make sure the filtered candidate list is unique.
        filtered_candidates = _unique_float_list(filtered_candidates, DECIMALS)

        # As soon as enough candidates exist, select them spread over the full range.
        if len(filtered_candidates) >= int(n):
            return _select_evenly_spread(filtered_candidates, int(n))

        # Otherwise densify again.
        density_factor += 1


def _in_range(val, lo, hi, tol):
    """
    Check whether a value belongs to a closed interval with tolerance.

    Args:
        val: float
            value to test
        lo: float
            lower bound
        hi: float
            upper bound
        tol: float
            tolerance for numeric robustness
        return: bool
            True if val is inside the interval up to tolerance
    """
    return (float(lo) - float(tol)) <= float(val) <= (float(hi) + float(tol))


def _list_ood_feature_names(combo, train_ranges):
    """
    List split-defining features whose values fall outside the train interval.

    Names match cfg train / test_ood_ranges keys: force_x, force_y, force_z, lengths, widths.

    Args:
        combo: tuple
            combo in the form ((fx, fy, fz), L, W)
        train_ranges: dict
            dict with keys fx, fy, fz, L, W mapping to (min, max)
        return: list of str
            OOD feature names in axis order
    """
    (fx, fy, fz), L, W = combo
    names = []
    if not _in_range(fx, train_ranges["fx"][0], train_ranges["fx"][1], TOL):
        names.append("force_x")
    if not _in_range(fy, train_ranges["fy"][0], train_ranges["fy"][1], TOL):
        names.append("force_y")
    if not _in_range(fz, train_ranges["fz"][0], train_ranges["fz"][1], TOL):
        names.append("force_z")
    if not _in_range(L, train_ranges["L"][0], train_ranges["L"][1], TOL):
        names.append("lengths")
    if not _in_range(W, train_ranges["W"][0], train_ranges["W"][1], TOL):
        names.append("widths")
    return names


def _count_ood_features(combo, train_ranges):
    """
    Count how many of the five split-defining features are outside the train interval.

    Args:
        combo: tuple
            combo in the form ((fx, fy, fz), L, W)
        train_ranges: dict
            dict mapping feature name to (min, max)
        return: int
            number of OOD features in the combo
    """
    return len(_list_ood_feature_names(combo, train_ranges))


def _split_target_counts(cfg):
    """
    Compute train / val / test targets from fractions relative to the total dataset.

    Args:
        cfg: dict
            yaml config
        return: tuple
            (n_train, n_val, n_test)
    """
    val_fraction = float(cfg["val_fraction"])
    test_fraction = float(cfg["test_fraction"])
    n_train = _train_count(cfg)
    total_count = int(round(n_train / (1.0 - val_fraction - test_fraction)))
    n_val = int(round(val_fraction * total_count))
    n_test = int(round(test_fraction * total_count))
    return n_train, n_val, n_test


def _varying_axis_count(cfg):
    """
    Count how many train axes actually vary.

    Args:
        cfg: dict
            yaml config
    return: int
        number of train axes that are not constant
    """
    t = cfg["train"]
    axis_configs = [t["force_x"], t["force_y"], t["force_z"], t["lengths"], t["widths"]]
    varying_axes = 0  # counter
    for axis_cfg in axis_configs:
        if not _is_constant_range(axis_cfg["start"], axis_cfg["stop"]):  # is not constant
            varying_axes += 1
    # Keep at least 1 to avoid invalid roots downstream.
    return max(1, varying_axes)


def _guess_count_values_per_axis(target_count, varying_axes):
    """
    Choose a per-axis guess that can later be scaled up if needed.

    Args:
        target_count: int
            target number of final samples
        varying_axes: int
            number of effective varying axes
        return: int
            initial per-axis guess
    """
    return max(1, int(ceil(int(target_count) ** (1.0 / int(varying_axes)))))


def _train_axis_values(cfg):
    """
    Return per-axis training lists.

    Important:
    - Constant axes return a single value only.
    - Therefore the train Cartesian product is already unique by construction.

    Args:
        cfg: dict
            yaml config
        return: tuple
            (fx, fy, fz, lengths, widths)
    """
    # Read the train section once.
    t = cfg["train"]

    # Build the five train axes.
    fx = _build_unique_axis_values(t["force_x"]["start"], t["force_x"]["stop"], t["force_x"]["n"], "force_x")
    fy = _build_unique_axis_values(t["force_y"]["start"], t["force_y"]["stop"], t["force_y"]["n"], "force_y")
    fz = _build_unique_axis_values(t["force_z"]["start"], t["force_z"]["stop"], t["force_z"]["n"], "force_z")
    lengths = _build_unique_axis_values(t["lengths"]["start"], t["lengths"]["stop"], t["lengths"]["n"], "lengths")
    widths = _build_unique_axis_values(t["widths"]["start"], t["widths"]["stop"], t["widths"]["n"], "widths")

    # Return the five axes.
    return fx, fy, fz, lengths, widths


def _train_ranges(cfg):
    """
    Return the min / max train interval of each split-defining feature.

    Args:
        cfg: dict
            yaml config
        return: dict
            dict mapping feature name to (min, max)
    """
    # Read the train section.
    t = cfg["train"]

    # Return the intervals used to define in-distribution vs out-of-distribution.
    return {"fx": (t["force_x"]["start"], t["force_x"]["stop"]),
            "fy": (t["force_y"]["start"], t["force_y"]["stop"]),
            "fz": (t["force_z"]["start"], t["force_z"]["stop"]),
            "L": (t["lengths"]["start"], t["lengths"]["stop"]),
            "W": (t["widths"]["start"], t["widths"]["stop"])}


def _validate_test_ood_ranges(cfg):
    """
    Validate that each configured test OOD range is fully outside the corresponding train range.
    This keeps the semantics sharp: values coming from test_ood_ranges must really be out-of-distribution.

    Args:
        cfg: dict
            yaml config
        return: None
            raises ValueError if an OOD range overlaps the train range
    """
    # Read the relevant config sections.
    t = cfg["train"]
    ood = cfg["test_ood_ranges"]
    # Describe the five axes once.
    axis_names = ["force_x", "force_y", "force_z", "lengths", "widths"]
    # Validate one axis at a time.
    for axis_name in axis_names:
        # Read the train and OOD xinterval.
        train_start = float(t[axis_name]["start"])
        train_stop = float(t[axis_name]["stop"])
        ood_start = float(ood[axis_name]["start"])
        ood_stop = float(ood[axis_name]["stop"])

        # Convert both to ordered intervals.
        train_lo = min(train_start, train_stop)
        train_hi = max(train_start, train_stop)
        ood_lo = min(ood_start, ood_stop)
        ood_hi = max(ood_start, ood_stop)

        # Accept only intervals fully on the left or fully on the right.
        is_fully_left = ood_hi < (train_lo - TOL)
        is_fully_right = ood_lo > (train_hi + TOL)

        # Reject any overlap.
        if not (is_fully_left or is_fully_right):
            raise ValueError(f"test_ood_ranges['{axis_name}'] overlaps the train range.")


def build_train(cfg):
    """
    Training is the simplest split:
    take the full Cartesian product of the unique train-axis values.

    Important:
    - No combo-level deduplication is needed anymore.
    - The train split is already unique by construction.

    Args:
        cfg: dict
            yaml config
        return: list of tuple
            cartesian product as list of tuples: ((fx, fy, fz), L, W)
    """
    fx, fy, fz, lengths, widths = _train_axis_values(cfg)
    forces = list(product(fx, fy, fz))
    return list(product(forces, lengths, widths))


def build_val(cfg):
    """
    Validation:
    values stay inside the training ranges, but are different from the exact training grid.

    Important:
    - Candidate values are generated in-range and excluding train values.
    - Selection is deterministic and spread over the admissible ordered list.
    - The final val subset is selected spread over the candidate combo list,
      instead of taking the first target_val_count entries.

    Args:
        cfg: dict
            yaml config
        return: list of tuple
            validation combos in the form ((fx, fy, fz), L, W)
    """
    # Read the train section once.
    t = cfg["train"]
    fx_tr, fy_tr, fz_tr, L_tr, W_tr = _train_axis_values(cfg)
    _, target_val_count, _ = _split_target_counts(cfg)
    # Count how many axes can actually create alternative in-range values.
    varying_axes = _varying_axis_count(cfg)
    # Guess a per-axis size and increase it only if needed.
    base_count = _guess_count_values_per_axis(target_val_count, varying_axes)
    # Start from scale 1.
    scale_factor = 1
    # Keep increasing density until the candidate validation pool is large enough.
    while True:
        # Build the per-axis target count for this iteration.
        n_per_axis = max(1, base_count * scale_factor)
        # Build in-range off-grid values for each axis.
        fx = vals_excluding(t["force_x"]["start"], t["force_x"]["stop"], n_per_axis, fx_tr)
        fy = vals_excluding(t["force_y"]["start"], t["force_y"]["stop"], n_per_axis, fy_tr)
        fz = vals_excluding(t["force_z"]["start"], t["force_z"]["stop"], n_per_axis, fz_tr)
        lengths = vals_excluding(t["lengths"]["start"], t["lengths"]["stop"], n_per_axis, L_tr)
        widths = vals_excluding(t["widths"]["start"], t["widths"]["stop"], n_per_axis, W_tr)
        # Build the full candidate grid.
        forces = list(product(fx, fy, fz))
        candidate_combos = list(product(forces, lengths, widths))
        # As soon as there are enough candidates, pick a spread deterministic subset.
        if len(candidate_combos) >= target_val_count:
            return _select_evenly_spread(candidate_combos, target_val_count)
        # Otherwise increase density and try again.
        scale_factor += 1


def _test_axis_values(id_start, id_stop, ood_start, ood_stop, n_axis, train_vals):
    """
    Generate the candidate values for one test axis.

    The returned dict contains:
    - "id": in-distribution values inside the train interval, but off the exact train grid
    - "ood": out-of-distribution values coming from the configured OOD interval

    Args:
        id_start: float
            train-range start
        id_stop: float
            train-range stop
        ood_start: float
            OOD-range start
        ood_stop: float
            OOD-range stop
        n_axis: int
            target number of values for both the ID and the OOD part
        train_vals: list of float
            exact train-grid values for this axis
        return: dict
            {"id": list of float, "ood": list of float}
    """
    # Build the in-distribution off-grid values.
    id_values = vals_excluding(id_start, id_stop, n_axis, train_vals)
    # Build the OOD values directly from the OOD interval.
    # They are unique by construction at the axis level.
    ood_values = _build_unique_axis_values(ood_start, ood_stop, n_axis, "test_ood_axis")
    # Return the two groups.
    return {"id": id_values, "ood": ood_values}


def _build_test_bucket_candidates(axis_value_groups, n_ood):
    """
    Build all test candidates that have exactly n_ood OOD features.

    This is the main semantic fix for the test split:
    instead of only filtering "at least one OOD" after the fact, we build the buckets explicitly.

    Args:
        axis_value_groups: list of dict
            one dict per axis with keys "id" and "ood"
        n_ood: int
            exact number of OOD features required
        return: list of tuple
            combos with exactly n_ood OOD features
    """
    # Prepare the output list for this bucket.
    bucket_combos = []

    # The five split-defining axes are indexed in this order.
    axis_indices = [0, 1, 2, 3, 4]

    # Enumerate every mask saying which axes are OOD.
    for ood_axes in combinations(axis_indices, n_ood):
        # For each axis, use either the OOD values or the ID values according to the mask.
        fx_values = axis_value_groups[0]["ood"] if 0 in ood_axes else axis_value_groups[0]["id"]
        fy_values = axis_value_groups[1]["ood"] if 1 in ood_axes else axis_value_groups[1]["id"]
        fz_values = axis_value_groups[2]["ood"] if 2 in ood_axes else axis_value_groups[2]["id"]
        L_values = axis_value_groups[3]["ood"] if 3 in ood_axes else axis_value_groups[3]["id"]
        W_values = axis_value_groups[4]["ood"] if 4 in ood_axes else axis_value_groups[4]["id"]

        # Build the force part for this mask.
        forces = list(product(fx_values, fy_values, fz_values))

        # Build all combos for this exact-OOD mask.
        mask_combos = list(product(forces, L_values, W_values))

        # Append them to the current exact-k bucket.
        bucket_combos.extend(mask_combos)

    # Return all candidates with exactly n_ood OOD features.
    return bucket_combos


def _allocate_bucket_counts(total_count, bucket_capacities):
    """
    Allocate the requested test size across the exact-OOD buckets.

    Policy:
    1. give one sample to each non-empty bucket, when possible
    2. distribute the remaining samples round-robin across the buckets

    This guarantees the intended behaviour:
    we get samples with 1 OOD, 2 OOD, 3 OOD, ... whenever the target size allows it.

    Args:
        total_count: int
            requested test split size
        bucket_capacities: list of int
            number of available candidates in each exact-k bucket
        return: list of int
            number of samples to take from each bucket
    """
    # Start with zero allocations.
    counts = [0] * len(bucket_capacities)

    # Keep only buckets that actually contain candidates.
    active_bucket_indices = []
    for bucket_index, capacity in enumerate(bucket_capacities):
        if capacity > 0:
            active_bucket_indices.append(bucket_index)

    # Nothing to allocate.
    if total_count == 0:
        return counts

    # Remaining number of samples to assign.
    remaining = int(total_count)

    # First pass:
    # give one sample to each non-empty bucket, in order, as long as possible.
    for bucket_index in active_bucket_indices:
        if remaining == 0:
            break
        counts[bucket_index] = 1
        remaining -= 1

    # Second pass:
    # keep filling buckets in round-robin order until the target size is reached or capacities are exhausted.
    while remaining > 0:
        # Track whether this pass could still place something.
        placed_something = False

        # Visit buckets in deterministic order.
        for bucket_index in active_bucket_indices:
            if remaining == 0:
                break

            # Skip full buckets.
            if counts[bucket_index] >= bucket_capacities[bucket_index]:
                continue

            # Add one element to this bucket.
            counts[bucket_index] += 1
            remaining -= 1
            placed_something = True

        # If no bucket could accept more, stop.
        if not placed_something:
            break

    # Return the final allocations.
    return counts


def build_test(cfg):
    """
    Test:
    build explicit exact-k-OOD buckets, with k in {1, 2, 3, 4, 5},
    then deterministically select the requested number of samples across those buckets.

    Important:
    - This directly implements the intended behaviour written in "the idea is".
    - Every kept test sample has at least one OOD feature.
    - The split contains samples with 1 OOD feature, 2 OOD features, and so on, whenever the target size allows it.
    - Selection is deterministic and spread over each candidate bucket.

    Args:
        cfg: dict
            yaml config
        return: list of tuple
            test combos in the form ((fx, fy, fz), L, W)
    """
    # Make sure the configured OOD intervals are really outside the train intervals.
    _validate_test_ood_ranges(cfg)

    # Read the train section.
    t = cfg["train"]

    # Read the test OOD ranges.
    ood = cfg["test_ood_ranges"]

    # Read the exact train-axis values.
    fx_tr, fy_tr, fz_tr, L_tr, W_tr = _train_axis_values(cfg)

    # Read the target test size.
    _, _, target_test_count = _split_target_counts(cfg)

    # Handle the trivial case explicitly.
    if target_test_count == 0:
        return []

    # Use five axes here because the exact-OOD buckets always refer to the five split-defining features.
    base_count = _guess_count_values_per_axis(target_test_count, 5)

    # Start from scale 1.
    scale_factor = 1

    # Keep increasing density until enough exact-k-OOD candidates exist.
    while True:
        # Build the per-axis candidate size for this iteration.
        n_axis = max(1, base_count * scale_factor)

        # Build the ID / OOD candidate values for each axis.
        axis_value_groups = [
            _test_axis_values(t["force_x"]["start"], t["force_x"]["stop"],
                              ood["force_x"]["start"], ood["force_x"]["stop"], n_axis, fx_tr),

            _test_axis_values(t["force_y"]["start"], t["force_y"]["stop"],
                              ood["force_y"]["start"], ood["force_y"]["stop"], n_axis, fy_tr),

            _test_axis_values(t["force_z"]["start"], t["force_z"]["stop"],
                              ood["force_z"]["start"], ood["force_z"]["stop"], n_axis, fz_tr),

            _test_axis_values(t["lengths"]["start"], t["lengths"]["stop"],
                              ood["lengths"]["start"], ood["lengths"]["stop"], n_axis, L_tr),

            _test_axis_values(t["widths"]["start"], t["widths"]["stop"],
                              ood["widths"]["start"], ood["widths"]["stop"], n_axis, W_tr),
        ]

        # Build the five exact-OOD buckets.
        bucket_candidates = []
        bucket_capacities = []

        # Bucket 1 means exactly 1 OOD feature, ..., bucket 5 means exactly 5 OOD features.
        for n_ood in range(1, 6):
            # Build all candidates for this exact-k bucket.
            candidates = _build_test_bucket_candidates(axis_value_groups, n_ood)

            # Store the candidates.
            bucket_candidates.append(candidates)

            # Store the bucket size.
            bucket_capacities.append(len(candidates))

        # Decide how many samples to draw from each exact-k bucket.
        bucket_counts = _allocate_bucket_counts(target_test_count, bucket_capacities)

        # If the current density supports the full target size, perform the final deterministic selection.
        if sum(bucket_counts) == target_test_count:
            # Prepare the final test split.
            test_combos = []

            # Select deterministically inside each bucket.
            for bucket_index, bucket_count in enumerate(bucket_counts):
                # Skip empty selections.
                if bucket_count == 0:
                    continue

                # Select a spread subset inside this bucket.
                selected = _select_evenly_spread(bucket_candidates[bucket_index], bucket_count)

                # Append the selected combos.
                test_combos.extend(selected)

            # Optional internal consistency check:
            # every combo must have between 1 and 5 OOD features.
            train_ranges = _train_ranges(cfg)
            for combo in test_combos:
                n_ood = _count_ood_features(combo, train_ranges)
                if n_ood < 1 or n_ood > 5:
                    raise RuntimeError("Internal error while building the test split.")

            # Return the final exact-size test split.
            return test_combos

        # Otherwise increase density and try again.
        scale_factor += 1


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------

def run_simulations(config_path, verbose, skip_existing, modes):
    """
    1. load yaml
    2. build only the requested split parameter lists
    3. create output folders
    4. launch fea_simulation once per parameter combination

    Split construction is deterministic (no randomness): same config yields the same
    combo order and trajectory indices as enumerate(combos) in the simulation loop.

    Args:
        config_path: str
            path of config file
        verbose: bool
            printing control
        skip_existing: bool
            skip existing graphs if True
        modes: tuple or list
            requested splits, e.g. ("val", "test")
        only_json: bool
            if True, only build the test split, write and print test_ood_features.json;
            no FEA runs. Requires "test" in modes and a non-empty test split.
        return: None
            runs the simulations or writes JSON only
    """
    cfg = load_config(config_path)
    defaults = cfg["defaults"]
    output_root = cfg["output_root"]

    requested_modes = tuple(str(mode).lower() for mode in modes)
    valid_modes = {"train", "val", "test"}
    for mode in requested_modes:
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'. Valid modes are: train, val, test.")

    if cfg["only_json"]:
        if "test" not in requested_modes:
            raise ValueError("only_json requires 'test' in modes.")
        n_train, n_val, n_test = _split_target_counts(cfg)
        if n_test == 0:
            raise ValueError("only_json requires a non-zero test split (check test_fraction / train size).")
        test_combos = build_test(cfg)
        if len(test_combos) != n_test:
            raise RuntimeError(f"Test split has length {len(test_combos)}, expected {n_test}.")
        test_dir = Path(output_root) / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        train_ranges = _train_ranges(cfg)
        ood_by_traj = {
            str(idx): _list_ood_feature_names(combo, train_ranges)
            for idx, combo in enumerate(test_combos)
        }
        ood_path = test_dir / "test_ood_features.json"
        with open(ood_path, "w") as f:
            json.dump(ood_by_traj, f, indent=2)
        print(json.dumps(ood_by_traj, indent=2))
        return

    n_train, n_val, n_test = _split_target_counts(cfg)
    expected_counts = {"train": n_train, "val": n_val, "test": n_test}
    builders = {"train": build_train, "val": build_val, "test": build_test}

    splits = {}
    for mode in requested_modes:
        splits[mode] = builders[mode](cfg)
        if len(splits[mode]) != expected_counts[mode]:
            raise RuntimeError(f"Split '{mode}' has length {len(splits[mode])}, "
                               f"but the expected target is {expected_counts[mode]}.")

    if "test" in requested_modes and len(splits["test"]) > 0:
        test_dir = Path(output_root) / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        train_ranges = _train_ranges(cfg)
        ood_by_traj = {
            str(idx): _list_ood_feature_names(combo, train_ranges)
            for idx, combo in enumerate(splits["test"])
        }
        ood_path = test_dir / "test_ood_features.json"
        with open(ood_path, "w") as f:
            json.dump(ood_by_traj, f, indent=2)

    for mode in requested_modes:
        combos = splits[mode]
        dir_path = Path(output_root) / mode / "graphs"
        dir_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60 + f"  {mode.upper()}: {len(combos)} simulations (target {expected_counts[mode]})" + "=" * 60)

        for idx, ((fx, fy, fz), L, W) in tqdm(enumerate(combos), total=len(combos)):
            p = dict(defaults)
            p["L"] = L
            p["W"] = W
            p["fx"] = fx
            p["fy"] = fy
            p["fz"] = fz

            if _graph_filepath(p, dir_path).exists() and skip_existing:
                continue

            if verbose:
                print(f"\nSim {idx + 1}/{len(combos)}:  fx={fx:.4f}  fy={fy:.4f}  fz={fz:.4f}  L={L:.4f}  W={W:.4f}")

            fea_simulation(L=L, W=W,
                           D=defaults["D"],
                           NL=defaults["NL"], NW=defaults["NW"], ND=defaults["ND"],
                           elastic_params=(defaults["E"], defaults["nu"]),
                           density=defaults["rho"],
                           damping_params=(defaults["eta_m"], defaults["eta_k"]),
                           newmark_params=(defaults["alpha_m"], defaults["alpha_f"]),
                           initial_force_x=fx, initial_force_y=fy, initial_force_z=fz,
                           cutoff_time_factor=defaults["cutoff_time_factor"],
                           total_time=defaults["total_time"],
                           num_steps=defaults["num_steps"],
                           mode=mode,
                           output_root=output_root,
                           verbose=verbose)


if __name__ == "__main__":
    config = "conf/fem_generation/config_original.yaml"

    run_simulations(config, verbose=False, skip_existing=False, modes=("val", "test"))
