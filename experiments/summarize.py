import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .common import GROUPS, PROTOCOL_PATH, job_id, plan, read_json, sha256, table_plan


def checkpoint_entries(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    if manifest['protocol_sha256'] != sha256(PROTOCOL_PATH):
        raise ValueError('Checkpoint protocol hash mismatch')
    indexed = {}
    expected = {job_id(job): job for group in GROUPS for job in plan(group)}
    for entry in manifest['checkpoints']:
        key = job_id(entry)
        if key in indexed or key not in expected:
            raise ValueError(f'Duplicate or unexpected checkpoint: {key}')
        indexed[key] = entry
    rows, digests = [], {}
    for key, job in expected.items():
        entry = indexed.get(key)
        if entry is None:
            rows.append(dict(**job, status='missing', metrics={}))
            continue
        entry = dict(entry)
        for file_key, hash_key in [('path', 'sha256'), ('scalers', 'scalers_sha256')]:
            path = (manifest_path.parent / entry[file_key]).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()):
                raise ValueError('Checkpoint path outside its manifest directory')
            if not path.is_file():
                entry.update(status='missing', metrics={})
                break
            if path not in digests:
                digests[path] = sha256(path)
            if digests[path] != entry[hash_key]:
                raise ValueError(f'Artifact hash mismatch: {path}')
        rows.append(entry)
    return rows


def aggregate(entries):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['group'], entry['task'], entry['model']].append(entry)
    results = []
    for (group, task, model), rows in sorted(groups.items()):
        completed = [row for row in rows if row['status'] == 'completed']
        count = len(completed)
        record = dict(group=group, task=task, model=model, expected_runs=len(rows), completed_runs=count,
                      failed_runs=sum(row['status'] == 'failed' for row in rows),
                      missing_runs=sum(row['status'] == 'missing' for row in rows),
                      unit='fm' if task == 'radius' else 'MeV', metrics={})
        roles = sorted({key for row in completed for key in row['metrics'] if key.startswith('test')})
        for role in roles:
            record['metrics'][role] = {}
            for name in ('rmse', 'mae', 'bias'):
                values = [row['metrics'][role][name] for row in completed if role in row['metrics']]
                complete = len(values) == len(rows)
                record['metrics'][role][name] = dict(mean=statistics.mean(values) if complete else None,
                                                    sample_sd=statistics.stdev(values) if complete and len(values) > 1 else None)
        results.append(record)
    return results


def markdown_table(entries, number, split_seed=None):
    names = {
        'cv_mi_pu_gru': 'CV MI-PU-GRU', 'cv_am_pu_gru': 'CV AM-PU-GRU',
        'real_gru': 'Real GRU', 'cv_gru': 'CV GRU', 'mlp': 'MLP (same input)',
        'real_mi_pu_gru': 'Real MI-PU-GRU', 'real_am_pu_gru': 'Real AM-PU-GRU',
        'cv_pu_gru': 'CV PU-GRU', 'cv_mi_gru': 'CV MI-GRU', 'cv_am_gru': 'CV AM-GRU',
        'cv_mi_tanh': 'CV MI (Tanh)', 'cv_am_tanh': 'CV AM (Tanh)',
        'cv_mi_silu': 'CV MI (SiLU)', 'cv_am_silu': 'CV AM (SiLU)',
    }
    jobs = [job for job in table_plan(number) if split_seed is None or job['split_seed'] == split_seed]
    expected = {job_id(job) for job in jobs}
    indexed = {job_id(entry): entry for entry in entries if job_id(entry) in expected}
    entries = [indexed.get(job_id(job), dict(**job, status='missing', metrics={})) for job in jobs]
    rows = {(row['task'], row['model']): row for row in aggregate(entries)}
    radius = number == 2
    columns = [('radius', 'test_interp', metric) for metric in ('rmse', 'mae', 'bias')] if radius else [
        ('mass_i', 'test_interp', 'rmse'), ('mass_ii', 'test_interp', 'rmse'), ('mass_ii', 'test_temporal', 'rmse')]
    headings = ['RMSE', 'MAE', 'Bias'] if radius else ['Mass I: interpolation', 'Mass II: interpolation', 'Mass II: temporal']
    lines = [f"Table {number}: mean ± sample standard deviation ({'fm' if radius else 'MeV'}).",
             '', '| Model | ' + ' | '.join(headings) + ' |', '| --- | ---: | ---: | ---: |']
    for model in dict.fromkeys(job['model'] for job in jobs):
        cells = []
        for task, role, metric in columns:
            value = rows.get((task, model), {}).get('metrics', {}).get(role, {}).get(metric, {})
            mean, sd = value.get('mean'), value.get('sample_sd')
            precision = 6 if radius else 4
            cells.append('incomplete' if mean is None or sd is None else f'{mean:.{precision}f} ± {sd:.{precision}f}')
        lines.append('| ' + names[model] + ' | ' + ' | '.join(cells) + ' |')
    completed = sum(entry['status'] == 'completed' for entry in entries)
    failed = sum(entry['status'] == 'failed' for entry in entries)
    missing = len(expected) - len(entries) + sum(entry['status'] == 'missing' for entry in entries)
    lines.extend(['', f'Runs: {completed}/{len(expected)} completed; {failed} failed; {missing} missing.'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=Path, default=Path('checkpoints/manifest.json'))
    parser.add_argument('--output-root', type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--group', choices=GROUPS)
    selection.add_argument('--table', type=int, choices=(1, 2, 3))
    parser.add_argument('--format', choices=('json', 'markdown'), default='json')
    parser.add_argument('--split-seed', type=int)
    args = parser.parse_args()
    if args.format == 'markdown' and args.table is None:
        parser.error('--format markdown requires --table')
    if args.output_root is None:
        entries = checkpoint_entries(args.checkpoints)
    else:
        entries = []
        for group in ([args.group] if args.group else GROUPS):
            for job in plan(group):
                path = args.output_root / 'runs' / job_id(job) / 'result.json'
                result = read_json(path) if path.exists() else dict(status='missing', metrics={})
                entries.append(dict(**job, status=result['status'], metrics=result.get('metrics', {})))
    entries = [entry for entry in entries if (args.group is None or entry['group'] == args.group)
               and (args.split_seed is None or entry['split_seed'] == args.split_seed)]
    if args.table:
        selected = {job_id(job) for job in table_plan(args.table)}
        entries = [entry for entry in entries if job_id(entry) in selected]
    if args.format == 'markdown':
        print(markdown_table(entries, args.table, args.split_seed))
    else:
        print(json.dumps(aggregate(entries), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
