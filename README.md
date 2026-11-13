# Product units in gated recurrent networks enhance prediction of nuclear masses and charge radii

PyTorch implementation accompanying the paper and extending our [ICCS 2026 study](#citation).

The repository includes real and complex MI-PU-GRU and AM-PU-GRU models, baselines, matched controls, experiment scripts, and 322 checkpoints covering Tables 1–3. Nuclear datasets are not included.

## Quick start

Python 3.12; PyTorch 2.7.1, NumPy 1.26.4, pandas 2.3.3, and scikit-learn 1.8.0.

```bash
python -m pip install -r requirements.txt
python -B example.py
```

`example.py` runs forward and backward passes with synthetic inputs. Models accept standardized real tensors `(batch, 5, 3)` and return `(batch, 1)`: four measured neighbors followed by the target, with channels `Z`, `N`, and the observable or theoretical prior. Radius inputs contain no mass features.

## Experiments

| Task | Partition | Split seeds | Training seeds | Checkpoint selection |
| --- | --- | --- | --- | --- |
| Mass I | 70/15/15 train/validation/test | 1, 2, 3 | 3, 4, 5 | Minimum validation loss |
| Mass II | 95/5 train/interpolation test; separate temporal test | 1, 2, 3 | 3, 4, 5 | Epoch 3000 |
| Radius | 70/15/15 train/validation/test | 0 | 0–4 | Minimum validation loss |

Tables 1 and 3 use the same inputs and scalers for each split, with nine runs per model and task. Table 3 includes both Tanh and SiLU replacements in Mass I and Mass II.

Prepare data from your local compilation and fixed prior tables:

```bash
python -m experiments.prepare --task mass_i --source all.csv --prior WS4_processed.csv
python -m experiments.prepare --task mass_ii --source all.csv --prior WS4_processed.csv
python -m experiments.prepare --task radius --source AME_2020_all.csv --prior WS_rch.txt
python -m experiments.run --group main_mass --device cuda --execute
python -m experiments.run --group additional_mass --device cuda --execute
python -m experiments.run --group radius --device cuda --execute
```

Mass CSV columns: `Z`, `N`, `mass excess [keV]`, `mass excess from systematics` (`N` for measured), and `discovery`. Radius CSV columns: `Z`, `N`, `radius [fm]`. WS4 columns: `z`, `a`, `mass_excess_kev`; WS* rows: `A Z radius_fm`.

Omit `--execute` to list runs. Use `python -m experiments.train --help` for individual models and seeds. Frozen index plans preserve the original neighborhood tie order across NumPy platforms without including nuclear measurements.

## Checkpoints

`checkpoints/manifest.json` indexes 162 Table 1 checkpoints, 70 Table 2 checkpoints, and 90 additional Table 3 checkpoints, with seeds, selected epochs, scalers, hashes, and metrics. Table 3 reuses the MI-PU-GRU, AM-PU-GRU, and CV GRU checkpoints from Table 1.

```bash
python -m experiments.summarize
python -m experiments.summarize --table 3 --format markdown
python -m experiments.evaluate --table 3 --device cuda
python -m experiments.evaluate --group radius --device cuda
python predict.py --checkpoint checkpoints/radius/split_0/cv_mi_pu_gru/seed_0.pth --inputs sequences.npy --output predictions.npy
```

The prediction command accepts unstandardized `(batch, 5, 3)` arrays and applies the stored scalers. Use keV for mass inputs and fm for radius inputs; outputs are MeV and fm, respectively. Complex PU logarithms use +π on the negative real axis for mass models and −π for radius models, matching their trained weights.

`experiments.evaluate` uses prepared data and saves test predictions under `outputs/evaluations`. Use `python -m experiments.analyze --help` for paired radius errors and influence analysis. Run `python -m experiments.benchmark --output outputs/radius_cost.json` for the paper's parameter, linear-FLOP, and GPU-latency measurements.

## Code layout

- [pu_gru/real.py](pu_gru/real.py): real-valued layers and recurrent cells.
- [pu_gru/complex.py](pu_gru/complex.py): complex-valued layers and recurrent cells.
- [pu_gru/models.py](pu_gru/models.py): sequence models and output layers.
- [experiments/](experiments/): data preparation, training, and analysis.

## Tests

Run the synthetic tests with:

```bash
python -B -m unittest discover -v
```

## Citation

If you use these architectures, please cite:

Ziyuan Li, Paulo S. A. Freitas, John W. Clark, and Babette Dellen. **Product Units in Gated Recurrent Units Improve Nuclear-Mass Prediction.** In *Computational Science — ICCS 2026*, pp. 141–156, 2026. [DOI: 10.1007/978-3-032-29921-5_10](https://doi.org/10.1007/978-3-032-29921-5_10).

## License

[MIT License](LICENSE). Copyright (c) 2026 Ziyuan Li.
