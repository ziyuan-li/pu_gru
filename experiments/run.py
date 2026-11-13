import argparse
import json
from pathlib import Path

from .common import GROUPS, plan
from .train import train, validate_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--task')
    parser.add_argument('--model')
    parser.add_argument('--split-seed', type=int)
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    jobs = [job for job in plan(args.group) if all(value is None or job[key] == value for key, value in
            [('task', args.task), ('model', args.model), ('split_seed', args.split_seed), ('training_seed', args.training_seed)])]
    if not jobs:
        parser.error('No experiment matches the requested filters')
    print(json.dumps(jobs, indent=2))
    if args.execute:
        for job in jobs:
            validate_bundle(args.data_root / job['task'] / f"split_{job['split_seed']}", job)
        for job in jobs:
            train(job, args.data_root, args.output_root, args.device)


if __name__ == '__main__':
    main()
