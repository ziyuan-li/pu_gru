import argparse
import json
from pathlib import Path

import numpy as np


def paired_analysis(candidate_paths, comparator_paths=None, resamples=50000, seed=20260923):
    if len(candidate_paths) != 5 or (comparator_paths is not None and len(comparator_paths) != 5):
        raise ValueError('Expected five paired training runs')
    if resamples <= 0 or seed < 0:
        raise ValueError('Invalid bootstrap configuration')
    candidate, comparator = [], []
    coordinates, truth = None, None
    for index, path in enumerate(candidate_paths):
        with np.load(path, allow_pickle=False) as data:
            if coordinates is None:
                coordinates, truth = data['ZN'].copy(), data['true'].copy()
            np.testing.assert_array_equal(data['ZN'], coordinates)
            np.testing.assert_array_equal(data['true'], truth)
            candidate.append(data['prediction'] - truth)
            prior = data['prior'].copy()
        if comparator_paths is None:
            comparator.append(prior - truth)
        else:
            with np.load(comparator_paths[index], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['ZN'], coordinates)
                np.testing.assert_array_equal(data['true'], truth)
                comparator.append(data['prediction'] - truth)
    candidate, comparator = np.asarray(candidate), np.asarray(comparator)
    if candidate.shape != (5, len(truth)) or comparator.shape != candidate.shape:
        raise ValueError('Expected one scalar prediction per nucleus and run')
    if not np.isfinite(candidate).all() or not np.isfinite(comparator).all():
        raise ValueError('Nonfinite paired errors')
    chains = np.unique(coordinates[:, 0])
    if len(chains) < 2:
        raise ValueError('At least two isotope chains are required')
    indices = [np.flatnonzero(coordinates[:, 0] == z) for z in chains]
    lengths = np.array([len(index) for index in indices])
    square = [np.array([np.square(errors[:, index]).sum(axis=1) for index in indices]) for errors in (candidate, comparator)]
    absolute = [np.array([np.abs(errors[:, index]).sum(axis=1) for index in indices]) for errors in (candidate, comparator)]
    generator = np.random.default_rng(seed)
    samples = []
    for start in range(0, resamples, 500):
        weights = generator.multinomial(len(chains), np.full(len(chains), 1.0 / len(chains)), size=min(500, resamples - start))
        denominator = (weights @ lengths)[:, None]
        rmse = [np.sqrt(weights @ value / denominator).mean(axis=1) for value in square]
        mae = [(weights @ value / denominator).mean(axis=1) for value in absolute]
        samples.append(np.column_stack([rmse[1] - rmse[0], mae[1] - mae[0]]))
    samples = np.vstack(samples)
    candidate_rmse = np.sqrt(np.mean(candidate ** 2, axis=1))
    comparator_rmse = np.sqrt(np.mean(comparator ** 2, axis=1))
    rmse_gain = comparator_rmse.mean() - candidate_rmse.mean()
    mae_gain = np.mean(np.abs(comparator)) - np.mean(np.abs(candidate))
    def omitted_gain(omitted):
        keep = np.ones(len(truth), dtype=bool)
        keep[omitted] = False
        return float(np.sqrt(np.mean(comparator[:, keep] ** 2, axis=1)).mean() -
                     np.sqrt(np.mean(candidate[:, keep] ** 2, axis=1)).mean())
    point_gain = np.mean(comparator ** 2 - candidate ** 2, axis=0)
    mass_number = coordinates.sum(axis=1)
    regions = {}
    for label, mask in [('A<60', mass_number < 60), ('60<=A<120', (mass_number >= 60) & (mass_number < 120)),
                        ('120<=A<180', (mass_number >= 120) & (mass_number < 180)), ('A>=180', mass_number >= 180)]:
        regions[label] = dict(nuclei=int(mask.sum()))
        for name, errors in [('candidate', candidate), ('comparator', comparator)]:
            values = np.sqrt(np.mean(errors[:, mask] ** 2, axis=1)) if mask.any() else None
            regions[label][name] = dict(rmse_mean=float(values.mean()), rmse_sample_sd=float(values.std(ddof=1))) if values is not None else None
    largest_gains = np.argsort(-point_gain)[:min(3, len(truth) - 1)]
    squared_error = np.sum(candidate ** 2, axis=0)
    largest_errors = np.argsort(-squared_error)[:3]
    return dict(nuclei=len(truth), isotope_chains=len(chains), training_runs=5, bootstrap_resamples=resamples,
                bootstrap_seed=seed, rmse_gain=float(rmse_gain), mae_gain=float(mae_gain),
                rmse_reduction_percent=float(100 * rmse_gain / comparator_rmse.mean()) if comparator_rmse.mean() else None,
                improved_seed_pairs=int(np.sum(candidate_rmse < comparator_rmse)),
                rmse_gain_interval=np.quantile(samples[:, 0], [0.025, 0.975]).tolist(),
                mae_gain_interval=np.quantile(samples[:, 1], [0.025, 0.975]).tolist(),
                improved_nuclei_fraction=float(np.mean(point_gain > 0)),
                mass_number_regions=regions,
                largest_gains_omitted=dict(nuclei=coordinates[largest_gains].tolist(), rmse_gain=omitted_gain(largest_gains)),
                largest_candidate_errors=dict(nuclei=coordinates[largest_errors].tolist(),
                    squared_error_share=float(squared_error[largest_errors].sum() / squared_error.sum()) if squared_error.sum() else 0.0),
                nucleus_squared_error_gains=[dict(Z=int(z), N=int(n), gain=float(gain))
                                            for (z, n), gain in zip(coordinates, point_gain)],
                leave_one_nucleus_out=[omitted_gain([i]) for i in range(len(truth))],
                leave_one_chain_out={str(int(z)): omitted_gain(index) for z, index in zip(chains, indices)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, nargs=5, required=True)
    parser.add_argument('--comparator', type=Path, nargs=5)
    parser.add_argument('--resamples', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20260923)
    args = parser.parse_args()
    print(json.dumps(paired_analysis(args.candidate, args.comparator, args.resamples, args.seed), indent=2))


if __name__ == '__main__':
    main()
