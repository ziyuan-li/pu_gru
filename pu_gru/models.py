import torch
from torch import nn

from .complex import (
    AMGRUCellC_NoPU,
    AMPUGRUCellC,
    ComplexGRUCell,
    ComplexLinear,
    MIGRUCellC_NoPU,
    MIPUGRUCellC,
    PUGRUCellC,
    ProductUnitLinearC,
)
from .controls import ComplexAffineSplit, replace_product_unit
from .real import AMPUGRUCell, GRUCell, MIPUGRUCell


MODEL_SPECS = {
    'real_gru': (False, 'GRU', 120),
    'real_mi_pu_gru': (False, 'MI-PU-GRU', 90),
    'real_am_pu_gru': (False, 'AM-PU-GRU', 75),
    'cv_gru': (True, 'GRU', 80),
    'cv_pu_gru': (True, 'PU-GRU', 80),
    'cv_mi_gru': (True, 'MI-GRU', 70),
    'cv_am_gru': (True, 'AM-GRU', 60),
    'cv_mi_pu_gru': (True, 'MI-PU-GRU', 60),
    'cv_am_pu_gru': (True, 'AM-PU-GRU', 50),
}

MATCHED_MODELS = {
    'cv_mi_tanh': ('cv_mi_pu_gru', 'tanh'),
    'cv_am_tanh': ('cv_am_pu_gru', 'tanh'),
    'cv_mi_silu': ('cv_mi_pu_gru', 'silu'),
    'cv_am_silu': ('cv_am_pu_gru', 'silu'),
}

MODEL_NAMES = (*MODEL_SPECS, 'mlp', *MATCHED_MODELS)
TASKS = ('mass_i', 'mass_ii', 'radius')
PHASE_CONVENTIONS = dict(mass_i='positive_pi', mass_ii='positive_pi', radius='negative_pi')


def validate_sequence(x, input_size):
    if x.ndim != 3 or x.shape[1:] != (5, input_size):
        raise ValueError(f'Expected input shape (batch, 5, {input_size})')
    if not x.is_floating_point():
        raise TypeError('Input sequences must be real floating-point tensors')


class SequenceRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, cell_type, complex_valued):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError('input_size and hidden_size must be positive')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.complex_valued = complex_valued
        cells = {
            'GRU': ComplexGRUCell,
            'PU-GRU': PUGRUCellC,
            'MI-GRU': MIGRUCellC_NoPU,
            'AM-GRU': AMGRUCellC_NoPU,
            'MI-PU-GRU': MIPUGRUCellC,
            'AM-PU-GRU': AMPUGRUCellC,
        } if complex_valued else {
            'GRU': GRUCell,
            'MI-PU-GRU': MIPUGRUCell,
            'AM-PU-GRU': AMPUGRUCell,
        }
        if cell_type not in cells:
            raise ValueError(f'Unsupported cell type: {cell_type}')
        self.rnn_cell = cells[cell_type](input_size, hidden_size)
        linear = ComplexLinear if complex_valued else nn.Linear
        self.fc = linear(hidden_size, 1)

    def forward(self, x):
        validate_sequence(x, self.input_size)
        x = x.to(dtype=self.fc.weight.dtype)
        h = x.new_ones((x.shape[0], self.hidden_size))
        for t in range(5):
            h = self.rnn_cell(x[:, t, :], h)
        output = self.fc(h)
        return output.real if self.complex_valued else output


class FlatMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError('hidden_size must be positive')
        self.hidden_size = hidden_size
        self.hidden1 = nn.Linear(15, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        validate_sequence(x, 3)
        x = x.to(dtype=self.hidden1.weight.dtype).flatten(start_dim=1)
        return self.fc(torch.tanh(self.hidden2(torch.tanh(self.hidden1(x)))))


def available_models(task):
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}')
    return MODEL_NAMES


def make_model(name, task='radius'):
    if name not in available_models(task):
        raise ValueError(f'Model {name!r} is not listed for task {task!r}')
    if name == 'mlp':
        return FlatMLP(92 if task == 'radius' else 187)
    base_name, activation = MATCHED_MODELS.get(name, (name, None))
    complex_valued, cell_type, mass_width = MODEL_SPECS[base_name]
    width = (mass_width + 1) // 2 if task == 'radius' else mass_width
    model = SequenceRegressor(3, width, cell_type, complex_valued)
    for module in model.modules():
        if isinstance(module, ProductUnitLinearC):
            module.negative_axis_phase = PHASE_CONVENTIONS[task]
    if activation is not None:
        model.rnn_cell.pu = ComplexAffineSplit(model.rnn_cell.pu, activation)
    return model


def make_matched_pair(path, task='radius', activation='tanh'):
    if path not in ('mi', 'am'):
        raise ValueError('path must be mi or am')
    if activation not in ('tanh', 'silu'):
        raise ValueError('activation must be tanh or silu')
    if f'cv_{path}_{activation}' not in available_models(task):
        raise ValueError('This matched control is not listed for the selected task')
    full = make_model(f'cv_{path}_pu_gru', task)
    return full, replace_product_unit(full, activation)


def real_parameter_count(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
