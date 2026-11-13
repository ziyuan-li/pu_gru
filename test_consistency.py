import cmath
import io
import math
import unittest

import torch
from torch.func import functional_call

from experiments.benchmark import linear_flops
from pu_gru import TASKS, available_models, make_model
from pu_gru.complex import (
    AMGRUCellC_NoPU,
    AMPUGRUCellC,
    ComplexGRUCell,
    MIGRUCellC_NoPU,
    MIPUGRUCellC,
    PUGRUCellC,
    ProductUnitLinearC,
)
from pu_gru.controls import ComplexAffineSplit
from pu_gru.real import AMPUGRUCell, GRUCell, MIPUGRUCell, ProductUnitLinear


CELL_CLASSES = {
    'real_gru': GRUCell,
    'real_mi_pu_gru': MIPUGRUCell,
    'real_am_pu_gru': AMPUGRUCell,
    'cv_gru': ComplexGRUCell,
    'cv_pu_gru': PUGRUCellC,
    'cv_mi_gru': MIGRUCellC_NoPU,
    'cv_am_gru': AMGRUCellC_NoPU,
    'cv_mi_pu_gru': MIPUGRUCellC,
    'cv_am_pu_gru': AMPUGRUCellC,
}

RADIUS_LINEAR_FLOPS = {
    'cv_mi_pu_gru': 195542,
    'cv_am_pu_gru': 194452,
    'real_gru': 115321,
    'real_mi_pu_gru': 107416,
    'real_am_pu_gru': 108947,
    'cv_gru': 207922,
    'cv_mi_gru': 160582,
    'cv_am_gru': 199442,
    'cv_pu_gru': 207922,
    'mlp': 20057,
}


def affine(layer, vector):
    weights = layer.weight.detach().tolist()
    bias = layer.bias.detach().tolist() if layer.bias is not None else [0] * len(weights)
    return [sum(w * x for w, x in zip(row, vector)) + b for row, b in zip(weights, bias)]


def add(left, right):
    return [a + b for a, b in zip(left, right)]


def multiply(left, right):
    return [a * b for a, b in zip(left, right)]


def mix(candidate, previous, gate):
    return [g * c + (1 - g) * p for c, p, g in zip(candidate, previous, gate)]


def sigmoid(vector):
    return [1 / (1 + math.exp(-x.real)) for x in vector]


def real_tanh(vector):
    return [math.tanh(x.real) for x in vector]


def reference_candidate(layer, vector):
    if isinstance(layer, ComplexAffineSplit):
        value = affine(layer, vector)
        fn = math.tanh if layer.activation == 'tanh' else lambda s: s / (1 + math.exp(-s))
        return [complex(fn(x.real), fn(x.imag)) for x in value]
    floor = [math.log1p(math.exp(t)) + layer.eps for t in layer.raw_theta.detach().tolist()[0]]
    if layer.weight.is_complex():
        phases = [cmath.phase(x) for x in vector]
        if layer.negative_axis_phase == 'negative_pi':
            phases = [-phase if phase == math.pi else phase for phase in phases]
        features = [complex(math.log(max(abs(x), t)), phase) for x, t, phase in zip(vector, floor, phases)]
        return [cmath.exp(x) for x in affine(layer, features)]
    logs = [math.log(max(x, t)) for x, t in zip(vector, floor)]
    mean = sum(logs) / len(logs)
    variance = sum((x - mean) ** 2 for x in logs) / len(logs)
    scale = layer.ln.weight.detach().tolist()
    offset = layer.ln.bias.detach().tolist()
    features = [
        (x - mean) / math.sqrt(variance + layer.ln.eps) * w + b
        for x, w, b in zip(logs, scale, offset)
    ]
    return [math.exp(x) for x in affine(layer, features)]


def reference_cell(name, cell, x, h):
    if name == 'real_am_pu_gru':
        width = len(h)
        x_main = affine(cell.x_to_main, x)
        h_gates = affine(cell.h_to_gates, h)
        gates = [
            sigmoid(add(x_main[k * width:(k + 1) * width], h_gates[k * width:(k + 1) * width]))
            for k in range(3)
        ]
        z, r, g = gates
        rh = multiply(r, h)
        h_parts = affine(cell.rh_to_cand, rh)
        additive = real_tanh(add(x_main[3 * width:], h_parts[:width]))
        interaction = [
            math.exp(value) for value in add(
                multiply(affine(cell.x_to_mi, x), h_parts[width:]), cell.mi_b.detach().tolist()
            )
        ]
        candidate = mix(reference_candidate(cell.pu, x + rh + interaction), additive, g)
        return mix(candidate, h, z)
    z = sigmoid(add(affine(cell.W_z, x), affine(cell.U_z, h)))
    r = sigmoid(add(affine(cell.W_r, x), affine(cell.U_r, h)))
    rh = multiply(r, h)
    if name in ('real_gru', 'cv_gru'):
        pre = add(affine(cell.W_h, x), affine(cell.U_h, rh))
        candidate = (
            [complex(math.tanh(v.real), math.tanh(v.imag)) for v in pre]
            if name == 'cv_gru' else real_tanh(pre)
        )
    elif name == 'cv_pu_gru':
        candidate = reference_candidate(cell.pu, x + rh)
    elif name in ('cv_mi_gru', 'cv_am_gru'):
        candidate = real_tanh(add(
            multiply(affine(cell.W_mi, x), affine(cell.U_mi, rh)), cell.mi_b.detach().tolist()
        ))
    else:
        interaction_module = cell.mi if name.startswith('cv_') else cell
        pre = add(
            multiply(affine(interaction_module.mi_Wx, x), affine(interaction_module.mi_Uh, rh)),
            interaction_module.mi_b.detach().tolist(),
        )
        exp = cmath.exp if name.startswith('cv_') else math.exp
        interaction = [exp(v) for v in pre]
        candidate = reference_candidate(cell.pu, x + rh + interaction)
    if name in ('cv_am_gru', 'cv_am_pu_gru'):
        g = sigmoid(add(affine(cell.W_g, x), affine(cell.U_g, h)))
        additive = real_tanh(add(affine(cell.W_add, x), affine(cell.U_add, rh)))
        candidate = mix(candidate, additive, g)
    return mix(candidate, h, z)


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cells_against_scalar_equations(self):
        for name, cell_class in CELL_CLASSES.items():
            for seed in range(5):
                with self.subTest(model=name, seed=seed):
                    torch.manual_seed(seed + 30)
                    cell = cell_class(3, 4)
                    with torch.no_grad():
                        for key, parameter in cell.named_parameters():
                            if key.endswith('raw_theta'):
                                parameter.add_(0.2 * torch.randn_like(parameter))
                            elif key.endswith('ln.weight'):
                                parameter.copy_(1 + 0.1 * torch.randn_like(parameter))
                            else:
                                parameter.copy_(0.2 * torch.randn_like(parameter))
                    dtype = torch.cfloat if name.startswith('cv_') else torch.float32
                    x = torch.randn(3, 3, dtype=dtype)
                    h = torch.randn(3, 4, dtype=dtype)
                    expected = torch.tensor([
                        reference_cell(name, cell, xi, hi) for xi, hi in zip(x.tolist(), h.tolist())
                    ], dtype=dtype)
                    torch.testing.assert_close(cell(x, h), expected, rtol=2e-5, atol=2e-6)

    def test_complete_sequences_against_scalar_equations(self):
        for name in available_models('radius'):
            if name == 'mlp':
                continue
            with self.subTest(model=name):
                torch.manual_seed(28)
                model = make_model(name)
                x = torch.randn(1, 5, 3)
                h = [complex(1, 0) if model.complex_valued else 1.0] * model.hidden_size
                base = name
                if name.endswith(('_tanh', '_silu')):
                    base = 'cv_am_pu_gru' if name.startswith('cv_am') else 'cv_mi_pu_gru'
                for step in x[0].tolist():
                    h = reference_cell(base, model.rnn_cell, step, h)
                expected = torch.tensor([[affine(model.fc, h)[0].real]])
                torch.testing.assert_close(model(x), expected, rtol=2e-5, atol=2e-6)

    def test_paper_widths(self):
        mass_widths = dict(zip(CELL_CLASSES, (120, 90, 75, 80, 80, 70, 60, 60, 50)))
        mass_widths.update({'cv_mi_tanh': 60, 'cv_am_tanh': 50, 'cv_mi_silu': 60, 'cv_am_silu': 50})
        for task in TASKS:
            for name in available_models(task):
                with self.subTest(task=task, model=name):
                    expected = mass_widths.get(name)
                    if name == 'mlp':
                        expected = 92 if task == 'radius' else 187
                    elif task == 'radius':
                        expected = math.ceil(expected / 2)
                    self.assertEqual(make_model(name, task).hidden_size, expected)

    def test_executed_linear_flops_match_paper(self):
        for name, expected in RADIUS_LINEAR_FLOPS.items():
            for batch in (1, 64):
                with self.subTest(model=name, batch=batch):
                    self.assertEqual(linear_flops(make_model(name), torch.randn(batch, 5, 3)), expected)

    def test_pu_and_cell_numerical_gradients(self):
        cases = [(name, cls(2, 2)) for name, cls in CELL_CLASSES.items()]
        cases.extend([('real_pu', ProductUnitLinear(3, 2)), ('cv_pu', ProductUnitLinearC(3, 2))])
        for name, module in cases:
            with self.subTest(model=name):
                torch.manual_seed(13)
                complex_valued = name.startswith('cv_')
                dtype = torch.cdouble if complex_valued else torch.double
                if name.endswith('_pu'):
                    values = [[0.3 + 0.2j, -1.4 + 0.8j, 2.1 - 0.3j]] if complex_valued else [[-0.3, 1.4, 2.1]]
                    args = (torch.tensor(values, dtype=dtype, requires_grad=True),)
                else:
                    values = [[0.3 + 0.2j, -1.4 + 0.8j]] if complex_valued else [[-0.3, 1.4]]
                    states = [[0.7 + 0.4j, 1.8 - 0.3j]] if complex_valued else [[0.7, 1.8]]
                    args = tuple(torch.tensor(v, dtype=dtype, requires_grad=True) for v in (values, states))
                keys, parameters = zip(*module.named_parameters())
                high_precision = tuple(
                    p.detach().to(torch.cdouble if p.is_complex() else torch.double).requires_grad_()
                    for p in parameters
                )

                def evaluate(*values):
                    count = len(args)
                    return functional_call(module, dict(zip(keys, values[count:])), values[:count])

                self.assertTrue(torch.autograd.gradcheck(
                    evaluate, args + high_precision, eps=1e-6, atol=2e-5, rtol=2e-4, fast_mode=False
                ))

    def test_batch_independence_and_state_dict_roundtrip(self):
        for task in TASKS:
            for name in available_models(task):
                with self.subTest(task=task, model=name):
                    torch.manual_seed(16)
                    model = make_model(name, task).eval()
                    x = torch.randn(64, 5, 3)
                    with torch.no_grad():
                        expected = model(x)
                        separate = torch.cat([model(row.unsqueeze(0)) for row in x])
                        torch.testing.assert_close(separate, expected, atol=2e-5, rtol=2e-5)
                        noncontiguous = x.transpose(1, 2).contiguous().transpose(1, 2)
                        torch.testing.assert_close(model(noncontiguous), expected)
                        changed = x.clone()
                        changed[0] += 0.25
                        torch.testing.assert_close(model(changed)[1:], expected[1:])
                        buffer = io.BytesIO()
                        torch.save(model.state_dict(), buffer)
                        buffer.seek(0)
                        restored = make_model(name, task).eval()
                        restored.load_state_dict(torch.load(buffer, weights_only=True))
                        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)

    def test_signed_and_zero_sequences(self):
        for task in TASKS:
            for name in available_models(task):
                for value in (0.0, -1.0, 1.0):
                    with self.subTest(task=task, model=name, value=value):
                        torch.manual_seed(0)
                        model = make_model(name, task)
                        x = torch.full((2, 5, 3), value, requires_grad=True)
                        y = model(x)
                        self.assertTrue(torch.isfinite(y).all())
                        y.square().mean().backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                        for p in model.parameters():
                            if p.grad is not None:
                                self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__':
    unittest.main()
