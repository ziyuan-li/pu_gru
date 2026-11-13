import math

import torch
from torch import nn
from torch.nn import functional as F


def to_complex(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x
    return torch.complex(x, torch.zeros_like(x))


class ComplexLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__(in_features, out_features, bias=bias)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat) * 0.05)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)


class ProductUnitLinearC(nn.Module):

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, theta_init: float = 1.0, eps: float = 1e-7, negative_axis_phase: str = 'positive_pi') -> None:
        super().__init__()
        if not math.isfinite(theta_init) or theta_init <= 0:
            raise ValueError('theta_init must be finite and positive')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')
        self.eps = eps
        if negative_axis_phase not in ('positive_pi', 'negative_pi'):
            raise ValueError('Invalid phase convention')
        self.negative_axis_phase = negative_axis_phase
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat)) if bias else None
        raw_init_real = theta_init + math.log(-math.expm1(-theta_init))
        self.raw_theta = nn.Parameter(torch.full((1, in_features), raw_init_real))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = to_complex(x)
        theta = F.softplus(self.raw_theta) + self.eps
        mag = x_c.abs()
        phase = torch.angle(x_c)
        if self.negative_axis_phase == 'negative_pi':
            phase = torch.where(phase == torch.pi, -phase, phase)
        mag_safe = torch.maximum(mag, theta)
        log_x = torch.complex(torch.log(mag_safe), phase)
        out = F.linear(log_x, self.weight, self.bias)
        return torch.exp(out)


class ComplexGatedCell(nn.Module):

    def _get_gated_activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        return pre_activation.real


class PUGRUCellC(ComplexGatedCell):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_z = ComplexLinear(input_size, hidden_size)
        self.U_z = ComplexLinear(hidden_size, hidden_size, bias=False)
        self.W_r = ComplexLinear(input_size, hidden_size)
        self.U_r = ComplexLinear(hidden_size, hidden_size, bias=False)
        self.pu = ProductUnitLinearC(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z_pre = self.W_z(x) + self.U_z(h_prev)
        r_pre = self.W_r(x) + self.U_r(h_prev)
        z = torch.sigmoid(self._get_gated_activation(z_pre))
        r = torch.sigmoid(self._get_gated_activation(r_pre))
        h_tilde = self.pu(torch.cat([x, r * h_prev], dim=-1))
        return (1 - z) * h_prev + z * h_tilde


class _MIBlock(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.mi_Wx = ComplexLinear(input_size, hidden_size, bias=False)
        self.mi_Uh = ComplexLinear(hidden_size, hidden_size, bias=False)
        self.mi_b = nn.Parameter(torch.zeros(hidden_size, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.mi_Wx(x) * self.mi_Uh(h) + self.mi_b)


class MIPUGRUCellC(ComplexGatedCell):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_z = ComplexLinear(input_size, hidden_size)
        self.U_z = ComplexLinear(hidden_size, hidden_size, bias=False)
        self.W_r = ComplexLinear(input_size, hidden_size)
        self.U_r = ComplexLinear(hidden_size, hidden_size, bias=False)
        self.mi = _MIBlock(input_size, hidden_size)
        self.pu = ProductUnitLinearC(input_size + hidden_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z_pre = self.W_z(x) + self.U_z(h_prev)
        r_pre = self.W_r(x) + self.U_r(h_prev)
        z = torch.sigmoid(self._get_gated_activation(z_pre))
        r = torch.sigmoid(self._get_gated_activation(r_pre))
        mi_pos = self.mi(x, r * h_prev)
        h_tilde = self.pu(torch.cat([x, r * h_prev, mi_pos], dim=-1))
        return (1 - z) * h_prev + z * h_tilde


class AMPUGRUCellC(ComplexGatedCell):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H = hidden_size
        self.W_z, self.U_z = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_r, self.U_r = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_add = ComplexLinear(input_size, H)
        self.U_add = ComplexLinear(H, H, bias=False)
        self.mi = _MIBlock(input_size, H)
        self.pu = ProductUnitLinearC(input_size + H + H, H)
        self.W_g = ComplexLinear(input_size, H)
        self.U_g = ComplexLinear(H, H, bias=False)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z_pre = self.W_z(x) + self.U_z(h_prev)
        r_pre = self.W_r(x) + self.U_r(h_prev)
        z = torch.sigmoid(self._get_gated_activation(z_pre))
        r = torch.sigmoid(self._get_gated_activation(r_pre))
        h_rh_prev = r * h_prev
        add_pre = self.W_add(x) + self.U_add(h_rh_prev)
        h_add = to_complex(torch.tanh(add_pre.real))
        mi_pos = self.mi(x, h_rh_prev)
        h_mul = self.pu(torch.cat([x, h_rh_prev, mi_pos], dim=-1))
        g_pre = self.W_g(x) + self.U_g(h_prev)
        g = torch.sigmoid(self._get_gated_activation(g_pre))
        h_tilde = g * h_mul + (1 - g) * h_add
        return (1 - z) * h_prev + z * h_tilde


class MIGRUCellC_NoPU(ComplexGatedCell):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H = hidden_size
        self.W_z, self.U_z = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_r, self.U_r = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_mi = ComplexLinear(input_size, H, bias=False)
        self.U_mi = ComplexLinear(H, H, bias=False)
        self.mi_b = nn.Parameter(torch.zeros(H, dtype=torch.cfloat))
        self.W_h, self.U_h = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self._get_gated_activation(self.W_z(x) + self.U_z(h_prev)))
        r = torch.sigmoid(self._get_gated_activation(self.W_r(x) + self.U_r(h_prev)))
        rh = r * h_prev
        mi_raw = self.W_mi(x) * self.U_mi(rh) + self.mi_b
        h_tilde = to_complex(torch.tanh(mi_raw.real))
        return (1 - z) * h_prev + z * h_tilde


class AMGRUCellC_NoPU(ComplexGatedCell):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H = hidden_size
        self.W_z, self.U_z = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_r, self.U_r = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_g, self.U_g = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_add, self.U_add = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_mi = ComplexLinear(input_size, H, bias=False)
        self.U_mi = ComplexLinear(H, H, bias=False)
        self.mi_b = nn.Parameter(torch.zeros(H, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self._get_gated_activation(self.W_z(x) + self.U_z(h_prev)))
        r = torch.sigmoid(self._get_gated_activation(self.W_r(x) + self.U_r(h_prev)))
        g = torch.sigmoid(self._get_gated_activation(self.W_g(x) + self.U_g(h_prev)))
        rh = r * h_prev
        add_pre = self.W_add(x) + self.U_add(rh)
        h_add = to_complex(torch.tanh(add_pre.real))
        mi_raw = self.W_mi(x) * self.U_mi(rh) + self.mi_b
        h_mul = to_complex(torch.tanh(mi_raw.real))
        h_tilde = g * h_mul + (1 - g) * h_add
        return (1 - z) * h_prev + z * h_tilde


class ComplexGRUCell(ComplexGatedCell):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H = hidden_size
        self.W_z, self.U_z = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_r, self.U_r = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))
        self.W_h, self.U_h = (ComplexLinear(input_size, H), ComplexLinear(H, H, bias=False))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z_pre = self.W_z(x) + self.U_z(h_prev)
        r_pre = self.W_r(x) + self.U_r(h_prev)
        z = torch.sigmoid(self._get_gated_activation(z_pre))
        r = torch.sigmoid(self._get_gated_activation(r_pre))
        h_lin = self.W_h(x) + self.U_h(r * h_prev)
        h_tilde = torch.complex(torch.tanh(h_lin.real), torch.tanh(h_lin.imag))
        return (1 - z) * h_prev + z * h_tilde
