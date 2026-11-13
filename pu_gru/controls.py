from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .complex import ProductUnitLinearC


class ComplexAffineSplit(nn.Module):
    def __init__(self, product_unit, activation):
        super().__init__()
        if not isinstance(product_unit, ProductUnitLinearC):
            raise TypeError('Expected a complex product-unit module')
        if activation not in ('tanh', 'silu'):
            raise ValueError('activation must be tanh or silu')
        self.activation = activation
        self.weight = nn.Parameter(product_unit.weight.detach().clone())
        self.bias = (
            nn.Parameter(product_unit.bias.detach().clone())
            if product_unit.bias is not None else None
        )

    def forward(self, x):
        value = F.linear(x, self.weight, self.bias)
        activation = torch.tanh if self.activation == 'tanh' else F.silu
        return torch.complex(activation(value.real), activation(value.imag))


def replace_product_unit(model, activation):
    if model.cell_type not in ('MI-PU-GRU', 'AM-PU-GRU') or not model.complex_valued:
        raise ValueError('Matched replacements require complex MI-PU-GRU or AM-PU-GRU')
    result = deepcopy(model)
    result.rnn_cell.pu = ComplexAffineSplit(result.rnn_cell.pu, activation)
    return result
