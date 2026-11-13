import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .common import PROTOCOL, sha256, write_json


def validate_frame(frame):
    frame = frame.copy()
    if len(frame) == 0 or frame.duplicated(['Z', 'N']).any():
        raise ValueError('Empty data or duplicate nuclides')
    if not np.isfinite(frame[['Z', 'N', 'value']].to_numpy()).all():
        raise ValueError('Nonfinite coordinates or labels')
    for key in ('Z', 'N'):
        if not np.equal(frame[key], np.round(frame[key])).all():
            raise ValueError('Nonintegral coordinates')
        frame[key] = frame[key].astype(int)
    frame['A'] = frame.Z + frame.N
    return frame.sort_values(['Z', 'N']).reset_index(drop=True)


def load_mass(path):
    frame = pd.read_csv(path)
    needed = ['Z', 'N', 'mass excess [keV]', 'mass excess from systematics', 'discovery']
    names = {}
    for name in needed:
        matches = [key for key in frame if key.strip() == name]
        if len(matches) != 1:
            raise ValueError(f'Missing or ambiguous column: {name}')
        names[matches[0]] = name
    frame = frame[list(names)].rename(columns=names)
    frame = frame.loc[frame['mass excess from systematics'].fillna('Y').str.strip().eq('N')].copy()
    frame = frame.rename(columns={'mass excess [keV]': 'value'})
    for key in ('Z', 'N', 'value'):
        frame[key] = pd.to_numeric(frame[key], errors='raise')
    frame = frame.loc[frame.value.notna() & (frame.Z >= 8) & (frame.N >= 8)].copy()
    frame['year'] = pd.to_numeric(frame.discovery, errors='coerce')
    return validate_frame(frame[['Z', 'N', 'value', 'year']])


def load_radius(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = [value.strip() for value in next(reader)]
        positions = {}
        for key in ('Z', 'N', 'radius [fm]'):
            if header.count(key) != 1:
                raise ValueError(f'Missing or ambiguous column: {key}')
            positions[key] = header.index(key)
        for row in reader:
            z, n = float(row[positions['Z']]), float(row[positions['N']])
            if z < 8 or n < 8:
                continue
            try:
                radius = float(row[positions['radius [fm]']])
            except (ValueError, TypeError):
                continue
            if np.isfinite(radius) and radius > 0:
                rows.append((z, n, radius))
    return validate_frame(pd.DataFrame(rows, columns=['Z', 'N', 'value']))


def load_prior(path, task):
    lookup = {}
    if task == 'radius':
        values = []
        for line in Path(path).read_text(encoding='ascii').splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
                a, z, value = int(fields[0]), int(fields[1]), float(fields[2])
                values.append((z, a - z, value))
    else:
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip().str.lower()
        values = [(row.z, row.a - row.z, row.mass_excess_kev) for row in frame.itertuples(index=False)]
    for z, n, value in values:
        if not np.isfinite([z, n, value]).all() or z != int(z) or n != int(n) or z < 0 or n < 0:
            raise ValueError('Invalid theoretical prior row')
        key = (int(z), int(n))
        if key in lookup or (task == 'radius' and value <= 0):
            raise ValueError('Duplicate or invalid theoretical prior')
        lookup[key] = float(value)
    if not lookup:
        raise ValueError('Empty theoretical prior')
    return lookup


def partitions(frame, task, seed):
    if task == 'mass_ii':
        reference = frame.loc[frame.year.isna() | (frame.year <= 2016)]
        temporal = frame.loc[frame.year > 2016]
        if len(temporal) == 0:
            raise ValueError('No temporal test targets')
        train, test = train_test_split(reference, test_size=0.05, random_state=seed)
        return dict(train=train, test_interp=test, test_temporal=temporal)
    train, holdout = train_test_split(frame, test_size=0.30, random_state=seed)
    val, test = train_test_split(holdout, test_size=0.50, random_state=seed)
    return dict(train=train, val=val, test_interp=test)


def sequences(targets, training, lookup, fixed_contexts=None):
    training = training.sort_values(['Z', 'N'])
    coordinates = training[['Z', 'N']].to_numpy()
    positions = {tuple(coordinate): i for i, coordinate in enumerate(coordinates)} if fixed_contexts is not None else None
    values, mass_numbers = training.value.to_numpy(), training.A.to_numpy()
    result = {key: [] for key in ('X_raw', 'y_raw', 'ZN', 'context_ZN', 'prior')}
    skipped = []
    for row in targets.sort_values(['Z', 'N']).itertuples(index=False):
        eligible = np.flatnonzero((mass_numbers <= row.A) &
                                 ~((coordinates[:, 0] == row.Z) & (coordinates[:, 1] == row.N)))
        if len(eligible) < 4:
            skipped.append([row.Z, row.N])
            continue
        distance = np.sqrt(np.square(coordinates[eligible] - [row.Z, row.N]).sum(axis=1))
        if fixed_contexts is None:
            nearest = eligible[np.argsort(distance, kind='quicksort')[:4]]
        else:
            context_keys = fixed_contexts[(row.Z, row.N)]
            nearest = np.array([positions[tuple(key)] for key in context_keys])
            if len(set(nearest)) != 4 or not set(nearest).issubset(eligible):
                raise ValueError('Invalid recorded context indices')
            chosen_distances = np.sqrt(np.square(coordinates[nearest] - [row.Z, row.N]).sum(axis=1))
            if np.any(np.diff(chosen_distances) < 0) or chosen_distances[-1] > np.partition(distance, 3)[3]:
                raise ValueError('Recorded contexts violate the nearest-neighbor rule')
        estimate = lookup[(row.Z, row.N)]
        context = np.column_stack([coordinates[nearest], values[nearest]])
        result['X_raw'].append(np.vstack([context, [row.Z, row.N, estimate]]))
        result['y_raw'].append([row.value])
        result['ZN'].append([row.Z, row.N])
        result['context_ZN'].append(coordinates[nearest])
        result['prior'].append(estimate)
    if not result['X_raw']:
        raise ValueError('No targets have four eligible training neighbors')
    return {key: np.asarray(value, dtype=np.int64 if key in ('ZN', 'context_ZN') else np.float64)
            for key, value in result.items()}, skipped


def recorded_contexts(frame, assigned, task, seed):
    path = Path(__file__).with_name('neighborhoods') / f'{task}_split_{seed}.json'
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding='utf-8'))
    coordinates = frame[['Z', 'N']].to_numpy(dtype='<i8')
    if hashlib.sha256(coordinates.tobytes()).hexdigest() != record['coordinates_sha256']:
        return {}
    result = {}
    for role, values in record['splits'].items():
        expected = assigned[role][['Z', 'N']].sort_values(['Z', 'N']).to_numpy()
        np.testing.assert_array_equal(coordinates[values['assigned']], expected)
        result[role] = {tuple(coordinates[index]): coordinates[context]
                        for index, context in zip(values['targets'], values['contexts'])}
    return result


def prepare_bundle(frame, lookup, task, seed, output, sources=None):
    if task not in PROTOCOL['tasks'] or seed < 0:
        raise ValueError('Invalid task or split seed')
    missing = set(map(tuple, frame[['Z', 'N']].to_numpy())) - set(lookup)
    if missing:
        raise ValueError(f'Theoretical prior missing {len(missing)} eligible nuclei')
    assigned = partitions(frame, task, seed)
    fixed = recorded_contexts(frame, assigned, task, seed)
    raw, skipped = {}, {}
    for name, targets in assigned.items():
        raw[name], skipped[name] = sequences(targets, assigned['train'], lookup, fixed.get(name))
    sx = StandardScaler().fit(raw['train']['X_raw'].reshape(-1, 15))
    sy = StandardScaler().fit(raw['train']['y_raw'])
    scalers = dict(x_mean=sx.mean_.tolist(), x_scale=sx.scale_.tolist(),
                   y_mean=sy.mean_.tolist(), y_scale=sy.scale_.tolist())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, values in raw.items():
        x = sx.transform(values['X_raw'].reshape(-1, 15)).reshape(-1, 5, 3)
        y = sy.transform(values['y_raw'])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite standardized sequences')
        np.savez_compressed(output / f'{name}.npz', X=x, y=y, **values)
    write_json(output / 'scalers.json', scalers)
    write_json(output / 'partitions.json', {name: dict(assigned=rows[['Z', 'N']].sort_values(['Z', 'N']).to_numpy().tolist(),
               retained=raw[name]['ZN'].tolist(), skipped=skipped[name]) for name, rows in assigned.items()})
    metadata = dict(task=task, split_seed=seed, **PROTOCOL['tasks'][task],
                    assigned_counts={name: len(rows) for name, rows in assigned.items()},
                    effective_counts={name: len(values['y_raw']) for name, values in raw.items()},
                    source_hashes=sources or {},
                    files={path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    write_json(output / 'metadata.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=PROTOCOL['tasks'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--split-seeds', type=int, nargs='+')
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    seeds = args.split_seeds if args.split_seeds is not None else PROTOCOL['radius' if args.task == 'radius' else 'main_mass']['split_seeds']
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error('Split seeds must be distinct nonnegative integers')
    frame = load_radius(args.source) if args.task == 'radius' else load_mass(args.source)
    lookup = load_prior(args.prior, args.task)
    for seed in seeds:
        target = args.output / args.task / f'split_{seed}'
        prepare_bundle(frame, lookup, args.task, seed, target,
                       dict(observations=sha256(args.source), prior=sha256(args.prior)))
        print(target)


if __name__ == '__main__':
    main()
