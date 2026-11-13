import ast
import io
import math
from pathlib import Path
import tokenize
import unittest

import torch
from torch.nn import functional as F

from pu_gru import (
    TASKS,
    available_models,
    make_matched_pair,
    make_model,
    real_parameter_count,
)
from pu_gru.complex import AMPUGRUCellC, ComplexGRUCell, ProductUnitLinearC
from pu_gru.real import AMPUGRUCell, GRUCell, ProductUnitLinear


RADIUS_COUNTS = {
    'real_gru': 11761,
    'real_mi_pu_gru': 11170,
    'real_am_pu_gru': 11296,
    'cv_gru': 10642,
    'cv_pu_gru': 10685,
    'cv_mi_gru': 10992,
    'cv_am_gru': 10262,
    'cv_mi_pu_gru': 10085,
    'cv_am_pu_gru': 10055,
    'mlp': 10121,
    'cv_mi_tanh': 10022,
    'cv_am_tanh': 10002,
    'cv_mi_silu': 10022,
    'cv_am_silu': 10002,
}


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_paper_parameter_counts(self):
        for name, count in RADIUS_COUNTS.items():
            with self.subTest(model=name):
                self.assertEqual(real_parameter_count(make_model(name)), count)
        self.assertEqual(real_parameter_count(make_model('cv_mi_pu_gru', 'mass_i')), 38165)
        self.assertEqual(real_parameter_count(make_model('mlp', 'mass_i')), 38336)
        model = make_model('cv_mi_gru')
        inactive = sum(
            p.numel() * 2 for name, p in model.named_parameters()
            if name.startswith(('rnn_cell.W_h.', 'rnn_cell.U_h.'))
        )
        self.assertEqual(inactive, 2730)

    def test_all_task_models_forward_backward_and_optimizer(self):
        for task in TASKS:
            for name in available_models(task):
                for seed in range(5):
                    with self.subTest(task=task, model=name, seed=seed):
                        torch.manual_seed(seed)
                        model = make_model(name, task)
                        x = torch.randn(8, 5, 3, requires_grad=True)
                        target = torch.randn(8, 1)
                        optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
                        result = model(x)
                        self.assertEqual(result.shape, (8, 1))
                        self.assertFalse(result.is_complex())
                        self.assertTrue(torch.isfinite(result).all())
                        F.mse_loss(result, target).backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                        for key, parameter in model.named_parameters():
                            inactive = name == 'cv_mi_gru' and key.startswith(
                                ('rnn_cell.W_h.', 'rnn_cell.U_h.')
                            )
                            if inactive:
                                self.assertIsNone(parameter.grad)
                            else:
                                self.assertIsNotNone(parameter.grad, key)
                                self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                        optimizer.step()
                        self.assertTrue(torch.isfinite(model(x.detach())).all())

    def test_five_steps_initial_state_and_complex_readout(self):
        for name in ('real_gru', 'cv_mi_pu_gru', 'cv_am_pu_gru'):
            with self.subTest(model=name):
                torch.manual_seed(11)
                model = make_model(name)
                x = torch.randn(3, 5, 3)
                states = []
                handle = model.rnn_cell.register_forward_pre_hook(
                    lambda module, args: states.append(tuple(t.detach().clone() for t in args))
                )
                output = model(x)
                handle.remove()
                self.assertEqual(len(states), 5)
                torch.testing.assert_close(states[0][1], torch.ones_like(states[0][1]))
                for t, (step_input, _) in enumerate(states):
                    torch.testing.assert_close(step_input.real, x[:, t])
                    if step_input.is_complex():
                        self.assertEqual(torch.count_nonzero(step_input.imag).item(), 0)
                last_state = model.rnn_cell(*states[-1])
                expected = (last_state @ model.fc.weight.T + model.fc.bias).real
                torch.testing.assert_close(output, expected)
                if model.complex_valued:
                    with torch.no_grad():
                        model.fc.bias.add_(100j)
                    torch.testing.assert_close(model(x), output)

    def test_complex_pu_powers_and_principal_phase(self):
        pu = ProductUnitLinearC(2, 3, theta_init=0.1)
        with torch.no_grad():
            pu.weight.copy_(torch.tensor([[1, 1], [1, -1], [0.5, 0]], dtype=torch.cfloat))
        x = torch.tensor([[2 + 0j, 3 + 0j], [-2 + 0j, 3 + 0j]])
        expected = torch.stack((x[:, 0] * x[:, 1], x[:, 0] / x[:, 1], x[:, 0].sqrt()), dim=1)
        torch.testing.assert_close(pu(x), expected, atol=1e-6, rtol=1e-6)
        branch = ProductUnitLinearC(1, 1)
        with torch.no_grad():
            branch.weight.fill_(1j)
        expected_branch = torch.exp(torch.tensor([[-math.pi + 1j * math.log(2)]]))
        torch.testing.assert_close(branch(torch.tensor([[-2.0]])), expected_branch)

    def test_floor_is_real_trainable_and_keeps_phase(self):
        pu = ProductUnitLinearC(3, 1)
        with torch.no_grad():
            pu.weight.fill_(0.2 + 0.1j)
        x = torch.tensor([[0j, -0.2 + 0j, 0.2j]], requires_grad=True)
        self.assertFalse(pu.raw_theta.is_complex())
        self.assertTrue(pu.raw_theta.requires_grad)
        self.assertEqual(pu.raw_theta.shape, (1, 3))
        floor = F.softplus(pu.raw_theta) + 1e-7
        torch.testing.assert_close(floor, torch.full_like(floor, 1.0 + 1e-7))
        features = torch.log(floor).to(torch.cfloat) + 1j * torch.tensor([[0, math.pi, math.pi / 2]])
        expected = torch.exp(features @ pu.weight.T + pu.bias)
        actual = pu(x)
        torch.testing.assert_close(actual, expected)
        actual.abs().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(pu.raw_theta.grad).all())
        self.assertGreater(torch.count_nonzero(pu.raw_theta.grad).item(), 0)

    def test_real_pu_normalizes_log_features(self):
        pu = ProductUnitLinear(3, 2)
        x = torch.tensor([[-2.0, 2.0, 4.0], [0.0, 3.0, 5.0]], requires_grad=True)
        log_x = torch.log(torch.maximum(x, F.softplus(pu.raw_theta) + 1e-7))
        mean = log_x.mean(dim=-1, keepdim=True)
        variance = (log_x - mean).square().mean(dim=-1, keepdim=True)
        normalized = (log_x - mean) / torch.sqrt(variance + pu.ln.eps)
        normalized = normalized * pu.ln.weight + pu.ln.bias
        expected = torch.exp(normalized @ pu.weight.T + pu.bias)
        torch.testing.assert_close(pu(x), expected)
        pu(x).sum().backward()
        self.assertGreater(torch.count_nonzero(pu.raw_theta.grad).item(), 0)

    def test_complex_gru_uses_real_gates_and_both_tanh_components(self):
        cell = ComplexGRUCell(1, 1)
        with torch.no_grad():
            for parameter in cell.parameters():
                parameter.zero_()
            cell.W_z.bias.fill_(-2 + 4j)
            cell.W_h.bias.fill_(0.4 + 0.7j)
        h = torch.tensor([[0.3 + 0.5j]])
        z = torch.sigmoid(torch.tensor(-2.0))
        candidate = torch.complex(torch.tanh(torch.tensor(0.4)), torch.tanh(torch.tensor(0.7)))
        expected = (1 - z) * h + z * candidate
        torch.testing.assert_close(cell(torch.ones(1, 1, dtype=torch.cfloat), h), expected)

    def test_complex_am_additive_path_has_zero_imaginary_part(self):
        cell = AMPUGRUCellC(1, 1)
        with torch.no_grad():
            for parameter in cell.parameters():
                parameter.zero_()
            cell.W_add.bias.fill_(0.4 + 10j)
        x = torch.ones(1, 1, dtype=torch.cfloat)
        h = torch.tensor([[0.3 + 0.5j]])
        candidate = 0.5 + 0.5 * torch.tanh(torch.tensor(0.4))
        expected = 0.5 * h + 0.5 * candidate
        torch.testing.assert_close(cell(x, h), expected)

    def test_real_gru_resets_before_recurrent_projection(self):
        cell = GRUCell(1, 2)
        with torch.no_grad():
            for parameter in cell.parameters():
                parameter.zero_()
            cell.W_r.bias.copy_(torch.logit(torch.tensor([0.2, 0.8])))
            cell.U_h.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        h = torch.ones(1, 2)
        expected = 0.5 * h + 0.5 * torch.tanh(torch.tensor([[0.8, 0.2]]))
        torch.testing.assert_close(cell(torch.ones(1, 1), h), expected)

    def test_real_am_fusion_gate_precedes_update_gate(self):
        cell = AMPUGRUCell(1, 1)
        with torch.no_grad():
            for parameter in cell.parameters():
                parameter.zero_()
            cell.x_to_main.bias.copy_(torch.tensor([
                math.log(0.2 / 0.8), 0.0, math.log(0.3 / 0.7), 0.4
            ]))
        h = torch.tensor([[0.6]])
        candidate = 0.3 + 0.7 * torch.tanh(torch.tensor(0.4))
        expected = 0.8 * h + 0.2 * candidate
        torch.testing.assert_close(cell(torch.ones(1, 1), h), expected)

    def test_matched_controls_share_initial_tensors_without_aliasing(self):
        for path in ('mi', 'am'):
            for activation in ('tanh', 'silu'):
                with self.subTest(path=path, activation=activation):
                    torch.manual_seed(5)
                    full, control = make_matched_pair(path, activation=activation)
                    full_state = full.state_dict()
                    control_state = control.state_dict()
                    self.assertEqual(set(full_state) - set(control_state), {'rnn_cell.pu.raw_theta'})
                    self.assertEqual(set(control_state) - set(full_state), set())
                    for key, value in control_state.items():
                        torch.testing.assert_close(value, full_state[key], rtol=0, atol=0)
                        self.assertNotEqual(value.data_ptr(), full_state[key].data_ptr())
                    self.assertEqual(
                        real_parameter_count(full) - real_parameter_count(control),
                        63 if path == 'mi' else 53,
                    )
                    torch.manual_seed(5)
                    preset = make_model(f'cv_{path}_{activation}')
                    for key, value in preset.state_dict().items():
                        torch.testing.assert_close(value, control_state[key], rtol=0, atol=0)
                    v = torch.randn(2, full.rnn_cell.pu.weight.shape[1], dtype=torch.cfloat)
                    affine = v @ full.rnn_cell.pu.weight.T + full.rnn_cell.pu.bias
                    fn = torch.tanh if activation == 'tanh' else F.silu
                    expected = torch.complex(fn(affine.real), fn(affine.imag))
                    torch.testing.assert_close(control.rnn_cell.pu(v), expected)

    def test_mlp_uses_all_fifteen_inputs(self):
        model = make_model('mlp')
        x = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) / 10
        h1 = torch.tanh(x.reshape(2, 15) @ model.hidden1.weight.T + model.hidden1.bias)
        h2 = torch.tanh(h1 @ model.hidden2.weight.T + model.hidden2.bias)
        torch.testing.assert_close(model(x), h2 @ model.fc.weight.T + model.fc.bias)

    def test_invalid_shapes_types_and_presets(self):
        for name in ('cv_mi_pu_gru', 'mlp'):
            model = make_model(name)
            for shape in ((2, 4, 3), (2, 5, 4), (2, 15)):
                with self.assertRaises(ValueError):
                    model(torch.ones(shape))
            for dtype in (torch.int64, torch.cfloat):
                with self.assertRaises(TypeError):
                    model(torch.ones(2, 5, 3, dtype=dtype))
        with self.assertRaises(ValueError):
            make_model('unknown')
        with self.assertRaises(ValueError):
            make_model('cv_gru', 'unknown')

    def test_python_sources_have_no_comments_or_docstrings(self):
        root = Path(__file__).resolve().parent
        paths = [*root.glob('*.py'), *(root / 'pu_gru').rglob('*.py'), *(root / 'experiments').rglob('*.py')]
        for path in paths:
            with self.subTest(file=path.name):
                source = path.read_text(encoding='utf-8')
                tokens = tokenize.generate_tokens(io.StringIO(source).readline)
                self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens))
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.assertIsNone(ast.get_docstring(node))


if __name__ == '__main__':
    unittest.main()
