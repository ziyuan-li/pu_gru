import argparse
from pathlib import Path

import numpy as np
import torch

from pu_gru.checkpoints import load_checkpoint
from .common import GROUPS, PROTOCOL, job_id, read_json, sha256, table_plan, write_json
from .train import evaluate, validate_bundle


def evaluate_checkpoint(path, data_root, output_root, device='cpu'):
    predictor = load_checkpoint(path, device)
    entry = predictor.metadata
    job = {key: entry[key] for key in ('group', 'task', 'model', 'split_seed', 'training_seed')}
    bundle = Path(data_root) / job['task'] / f"split_{job['split_seed']}"
    metadata = validate_bundle(bundle, job)
    scalers = read_json(bundle / 'scalers.json')
    for key in ('x_mean', 'x_scale', 'y_mean', 'y_scale'):
        np.testing.assert_allclose(scalers[key], getattr(predictor, key).cpu().numpy(), rtol=1e-12, atol=1e-12,
                                   err_msg='Dataset scalers differ from the checkpoint scalers')
    output = Path(output_root) / job_id(job)
    output.mkdir(parents=True, exist_ok=False)
    roles = [role for role in metadata['split_roles'] if role.startswith('test')]
    metrics = {role: evaluate(predictor.model, bundle, role, output, torch.device(device), job['task']) for role in roles}
    result = dict(job=job, checkpoint_sha256=sha256(path), data_sha256=sha256(bundle / 'metadata.json'),
                  metric_unit=metadata['metric_unit'], metrics=metrics)
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--checkpoint', type=Path)
    selection.add_argument('--group', choices=GROUPS)
    selection.add_argument('--table', type=int, choices=(1, 2, 3))
    parser.add_argument('--task', choices=PROTOCOL['tasks'])
    parser.add_argument('--model')
    parser.add_argument('--manifest', type=Path, default=Path('checkpoints/manifest.json'))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/evaluations'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.checkpoint:
        paths = [args.checkpoint]
    else:
        selected = {job_id(job) for job in table_plan(args.table)} if args.table else None
        paths = [args.manifest.parent / row['path'] for row in read_json(args.manifest)['checkpoints']
                 if (job_id(row) in selected if selected is not None else row['group'] == args.group)
                 and (args.task is None or row['task'] == args.task)
                 and (args.model is None or row['model'] == args.model)]
    if not paths:
        parser.error('No matching checkpoint')
    for path in paths:
        result = evaluate_checkpoint(path, args.data_root, args.output_root, args.device)
        print(f"{job_id(result['job'])}: {result['metrics']}", flush=True)


if __name__ == '__main__':
    main()
