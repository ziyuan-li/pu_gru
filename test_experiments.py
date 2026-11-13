from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.analyze import paired_analysis
from experiments.common import PROTOCOL, job_id, plan, table_plan, training_settings
from experiments.evaluate import evaluate_checkpoint
from experiments.prepare import partitions, prepare_bundle, sequences, validate_frame
from experiments.summarize import aggregate, checkpoint_entries, markdown_table
from experiments.train import batches, centralize, optimizer_for, train
from pu_gru import make_model, real_parameter_count
from pu_gru.checkpoints import load_checkpoint
from pu_gru.complex import ProductUnitLinearC


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.frame = validate_frame(pd.DataFrame([
            dict(Z=z, N=n, value=z * 123.0 + n * 17.0, year=2020 if n > 24 else 2010)
            for z in range(8, 16) for n in range(8, 28)]))
        cls.lookup = {(row.Z, row.N): row.value + 2.0 for row in cls.frame.itertuples()}
        for task in PROTOCOL['tasks']:
            prepare_bundle(cls.frame, cls.lookup, task, 0, cls.root / 'data' / task / 'split_0')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_partitions_contexts_and_scalers(self):
        for task in PROTOCOL['tasks']:
            assigned = partitions(self.frame, task, 0)
            keys = {role: set(map(tuple, rows[['Z', 'N']].to_numpy())) for role, rows in assigned.items()}
            for role in keys:
                for other in keys:
                    if role != other:
                        self.assertFalse(keys[role] & keys[other])
            if task == 'mass_ii':
                self.assertNotIn('val', assigned)
                reference_count = len(self.frame.loc[self.frame.year <= 2016])
                self.assertEqual(len(assigned['test_interp']), int(np.ceil(reference_count * .05)))
                self.assertTrue((assigned['test_temporal'].year > 2016).all())
            else:
                self.assertEqual([len(rows) for rows in assigned.values()], [112, 24, 24])
            folder = self.root / 'data' / task / 'split_0'
            for role in assigned:
                with np.load(folder / f'{role}.npz') as values:
                    for target, context in zip(values['ZN'], values['context_ZN']):
                        self.assertTrue(set(map(tuple, context)) <= keys['train'])
                        self.assertNotIn(tuple(target), set(map(tuple, context)))
                        self.assertTrue((context.sum(axis=1) <= target.sum()).all())
            with np.load(folder / 'train.npz') as values:
                scaler = json.loads((folder / 'scalers.json').read_text())
                np.testing.assert_allclose(scaler['x_mean'], values['X_raw'].reshape(-1, 15).mean(axis=0))
                np.testing.assert_allclose(scaler['y_mean'], values['y_raw'].mean(axis=0))

    def test_heldout_labels_do_not_enter_inputs(self):
        assigned = partitions(self.frame, 'mass_i', 0)
        original, _ = sequences(assigned['test_interp'], assigned['train'], self.lookup)
        changed = assigned['test_interp'].copy()
        changed['value'] += 10000
        updated, _ = sequences(changed, assigned['train'], self.lookup)
        np.testing.assert_array_equal(original['X_raw'], updated['X_raw'])
        np.testing.assert_allclose(updated['y_raw'] - original['y_raw'], 10000)

    def test_experiment_grid_and_selection(self):
        self.assertEqual(len(plan('main_mass')), 162)
        self.assertEqual(len(plan('additional_mass')), 90)
        self.assertEqual(len(plan('radius')), 70)
        for job in plan('main_mass') + plan('additional_mass'):
            self.assertIn(job['split_seed'], (1, 2, 3))
            self.assertIn(job['training_seed'], (3, 4, 5))
            settings = training_settings(job['task'], job['model'])
            if job['task'] == 'mass_ii':
                self.assertEqual(settings['selection'], 'final_epoch')
            if job['group'] == 'additional_mass':
                self.assertFalse(settings['gradient_centralization'])

    def test_table3_reuses_main_references(self):
        jobs = table_plan(3)
        self.assertEqual(len(jobs), 144)
        self.assertEqual(len({job_id(job) for job in jobs}), 144)
        references = {job_id(job) for job in plan('main_mass')
                      if job['model'] in ('cv_mi_pu_gru', 'cv_am_pu_gru', 'cv_gru')}
        self.assertEqual({job_id(job) for job in jobs if job['group'] == 'main_mass'}, references)
        self.assertEqual(len(references), 54)
        for task in ('mass_i', 'mass_ii'):
            for model in PROTOCOL['additional_mass']['models']:
                self.assertEqual(sum(job['task'] == task and job['model'] == model for job in jobs), 9)

    def test_gradient_centralization_and_decay_groups(self):
        model = make_model('cv_mi_pu_gru', 'radius')
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        bias_before = model.fc.bias.grad.clone()
        centralize(model)
        for name, parameter in model.named_parameters():
            if name.endswith('weight') and parameter.ndim > 1:
                torch.testing.assert_close(parameter.grad.mean(dim=1), torch.zeros_like(parameter.grad[:, 0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(model.fc.bias.grad, bias_before)
        optimizer = optimizer_for(model, training_settings('radius', 'cv_mi_pu_gru'))
        decay = {id(parameter): group['weight_decay'] for group in optimizer.param_groups for parameter in group['params']}
        for name, parameter in model.named_parameters():
            self.assertEqual(decay[id(parameter)], 1e-4 if name.endswith('weight') and parameter.ndim > 1 else 0.0)

    def test_phase_boundary_conventions(self):
        for phase, expected in [('positive_pi', -np.pi), ('negative_pi', np.pi)]:
            layer = ProductUnitLinearC(1, 1, negative_axis_phase=phase)
            with torch.no_grad():
                layer.weight.fill_(1j)
            actual = layer(torch.tensor([[-2.0]])).abs()
            torch.testing.assert_close(actual, torch.tensor([[np.exp(expected)]], dtype=torch.float32))
        self.assertEqual(make_model('cv_mi_pu_gru', 'radius').rnn_cell.pu.negative_axis_phase, 'negative_pi')
        self.assertEqual(make_model('cv_mi_pu_gru', 'mass_i').rnn_cell.pu.negative_axis_phase, 'positive_pi')

    def test_mass_ii_uses_final_state_without_validation(self):
        job = dict(group='main_mass', task='mass_ii', model='real_gru', split_seed=0, training_seed=3)
        output = self.root / 'mass_ii_training'
        with patch.dict(PROTOCOL, epochs=2), redirect_stdout(io.StringIO()):
            result = train(job, self.root / 'data', output)
        self.assertEqual(result['selected_epoch'], 2)
        self.assertEqual(result['selection'], 'final_epoch')
        self.assertEqual(set(result['metrics']), {'train', 'test_interp', 'test_temporal'})
        folder = output / 'runs' / job_id(job)
        selected = torch.load(folder / 'selected.pth', weights_only=True)
        final = torch.load(folder / 'final.pth', weights_only=True)
        for key in selected:
            torch.testing.assert_close(selected[key], final[key], atol=0, rtol=0)
        predictor = load_checkpoint(folder / 'selected.pth')
        with np.load(self.root / 'data/mass_ii/split_0/test_interp.npz') as values:
            with torch.no_grad():
                prediction = predictor(torch.tensor(values['X_raw'])).numpy().ravel()
        with np.load(folder / 'predictions_test_interp.npz') as values:
            np.testing.assert_allclose(prediction, values['prediction'], atol=1e-7)
        repeated = evaluate_checkpoint(folder / 'selected.pth', self.root / 'data', self.root / 'evaluation')
        self.assertEqual(repeated['metrics'], {role: values for role, values in result['metrics'].items() if role.startswith('test')})

    def test_resume_preserves_training_randomness(self):
        job = dict(group='main_mass', task='mass_i', model='real_gru', split_seed=0, training_seed=4)
        calls = 0
        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            yield from batches(*args)
        with patch.dict(PROTOCOL, epochs=3), redirect_stdout(io.StringIO()):
            train(job, self.root / 'data', self.root / 'continuous')
            with patch('experiments.train.batches', interrupted), self.assertRaises(KeyboardInterrupt):
                train(job, self.root / 'data', self.root / 'resumed')
            result = train(job, self.root / 'data', self.root / 'resumed')
        states = [torch.load(self.root / output / 'runs' / job_id(job) / 'final.pth', weights_only=True)
                  for output in ('continuous', 'resumed')]
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
        self.assertEqual(result['completed_epochs'], 3)

    def test_radius_threshold_freeze_and_smoke_isolation(self):
        job = dict(group='radius', task='radius', model='cv_mi_pu_gru', split_seed=0, training_seed=0)
        flags = []
        def tracked_optimizer(model, settings):
            optimizer = optimizer_for(model, settings)
            def tracked_step(optimizer, args, kwargs):
                flags.append(model.rnn_cell.pu.raw_theta.requires_grad)
            optimizer.register_step_pre_hook(tracked_step)
            return optimizer
        with patch('experiments.train.optimizer_for', tracked_optimizer), redirect_stdout(io.StringIO()):
            result = train(job, self.root / 'data', self.root / 'radius_training', smoke_epochs=6)
        self.assertEqual(result['status'], 'smoke')
        self.assertEqual(set(result['metrics']), {'train', 'val'})
        self.assertEqual(flags, [False] * 10 + [True] * 2)

    def test_failures_are_not_silently_excluded(self):
        job = dict(group='main_mass', task='mass_i', model='real_gru')
        records = [dict(**job, status='completed', metrics={'test_interp': dict(rmse=.2, mae=.1, bias=0)}),
                   dict(**job, status='failed', metrics={})]
        result = aggregate(records)[0]
        self.assertEqual(result['expected_runs'], 2)
        self.assertEqual(result['failed_runs'], 1)
        self.assertIsNone(result['metrics']['test_interp']['rmse']['mean'])

    def test_released_checkpoint_coverage_and_loading(self):
        root = Path(__file__).resolve().parent / 'checkpoints'
        entries = checkpoint_entries(root / 'manifest.json')
        self.assertEqual(len(entries), 322)
        self.assertEqual(sum(row['group'] == 'main_mass' for row in entries), 162)
        self.assertEqual(sum(row['group'] == 'additional_mass' for row in entries), 90)
        self.assertEqual(sum(row['group'] == 'radius' for row in entries), 70)
        self.assertEqual({row['path'] for row in entries}, {path.relative_to(root).as_posix() for path in root.rglob('*.pth')})
        for row in entries:
            with self.subTest(checkpoint=row['path']):
                self.assertEqual(row['status'], 'completed')
                self.assertEqual(row['completed_epochs'], 3000)
                self.assertEqual(row['selection'], PROTOCOL['tasks'][row['task']]['selection'])
                if row['task'] == 'mass_ii':
                    self.assertEqual(row['selected_epoch'], 3000)
                predictor = load_checkpoint(root / row['path'])
                self.assertEqual(real_parameter_count(predictor.model), row['real_parameter_components'])
                with torch.no_grad():
                    value = predictor(predictor.x_mean.reshape(1, 5, 3))
                self.assertEqual(value.shape, (1, 1))
                self.assertTrue(torch.isfinite(value).all())

    def test_missing_checkpoint_stays_in_summary(self):
        manifest_path = Path(__file__).resolve().parent / 'checkpoints/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        missing = manifest['checkpoints'].pop()
        with patch('experiments.summarize.read_json', return_value=manifest):
            entries = checkpoint_entries(manifest_path)
        self.assertEqual(len(entries), 322)
        target = next(row for row in aggregate(entries) if (row['task'], row['model']) == (missing['task'], missing['model']))
        self.assertEqual(target['missing_runs'], 1)
        self.assertIsNone(target['metrics']['test_interp']['rmse']['mean'])

    def test_table3_markdown_keeps_missing_runs(self):
        entries = [dict(**job, status='completed', metrics={
            role: dict(rmse=.2, mae=.1, bias=0)
            for role in PROTOCOL['tasks'][job['task']]['split_roles'] if role.startswith('test')})
            for job in table_plan(3)]
        text = markdown_table(entries, 3)
        self.assertIn('144/144 completed; 0 failed; 0 missing', text)
        self.assertIn('0.2000 ± 0.0000', text)
        self.assertIn('48/48 completed; 0 failed; 0 missing', markdown_table(entries, 3, split_seed=1))
        entries.pop()
        text = markdown_table(entries, 3)
        self.assertIn('143/144 completed; 0 failed; 1 missing', text)
        self.assertIn('| CV AM (SiLU) | 0.2000 ± 0.0000 | incomplete | incomplete |', text)

    def test_additional_mass_loader_and_final_selection(self):
        job = dict(group='additional_mass', task='mass_ii', model='cv_mi_silu', split_seed=0, training_seed=3)
        observed = []
        def tracked_batches(x, y, generator, loader, batch_size):
            observed.append(loader is not None)
            yield from batches(x, y, generator, loader, batch_size)
        with patch.dict(PROTOCOL, epochs=2), patch('experiments.train.batches', tracked_batches), redirect_stdout(io.StringIO()):
            result = train(job, self.root / 'data', self.root / 'additional_training')
        self.assertEqual(observed, [True, True])
        self.assertEqual(result['selected_epoch'], 2)
        self.assertNotIn('val', result['metrics'])
        folder = self.root / 'additional_training/runs' / job_id(job)
        selected, final = [torch.load(folder / name, weights_only=True) for name in ('selected.pth', 'final.pth')]
        for key in selected:
            torch.testing.assert_close(selected[key], final[key], atol=0, rtol=0)

    def test_paired_analysis_sign_and_chain_resampling(self):
        coordinates = np.array([[10, 12], [10, 14], [40, 50], [40, 52], [70, 90], [90, 130]])
        paths = []
        for seed in range(5):
            path = self.root / f'paired_{seed}.npz'
            np.savez(path, ZN=coordinates, true=np.zeros(6), prediction=np.full(6, 1.0), prior=np.full(6, 2.0))
            paths.append(path)
        result = paired_analysis(paths, resamples=100)
        self.assertEqual(result['rmse_gain'], 1.0)
        self.assertEqual(result['mae_gain'], 1.0)
        self.assertEqual(result['rmse_gain_interval'], [1.0, 1.0])
        self.assertEqual(result['improved_seed_pairs'], 5)
        self.assertEqual(result['rmse_reduction_percent'], 50.0)
        self.assertEqual(sum(region['nuclei'] for region in result['mass_number_regions'].values()), 6)
        self.assertEqual(result['largest_gains_omitted']['rmse_gain'], 1.0)


if __name__ == '__main__':
    unittest.main()
