import math

import torch
from torch import nn
from torch.nn import functional as F


def inv_softplus(x: float) -> float:
    return x + math.log(-math.expm1(-x))


class ProductUnitLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, theta_init: float = 1.0, eps: float = 1e-7):
        super().__init__()
        if not math.isfinite(theta_init) or theta_init <= 0:
            raise ValueError('theta_init must be finite and positive')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        raw_init = inv_softplus(theta_init)
        self.raw_theta = nn.Parameter(torch.full((1, in_features), raw_init))
        self.ln = nn.LayerNorm(in_features)
        nn.init.uniform_(self.weight, -0.1, 0.1)

    def forward(self, x):
        theta = F.softplus(self.raw_theta) + self.eps
        x_safe = torch.clamp_min(x, theta)
        log_x = self.ln(torch.log(x_safe))
        out = F.linear(log_x, self.weight, self.bias)
        return torch.exp(out)


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h_prev))
        return (1 - z) * h_prev + z * h_tilde


class MIPUGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mi_Wx = nn.Linear(input_size, hidden_size, bias=False)
        self.mi_Uh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mi_b = nn.Parameter(torch.zeros(hidden_size))
        self.pu = ProductUnitLinear(input_size + hidden_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))
        mi_pos = torch.exp(self.mi_Wx(x) * self.mi_Uh(r * h_prev) + self.mi_b)
        h_tilde = self.pu(torch.cat([x, r * h_prev, mi_pos], -1))
        return (1 - z) * h_prev + z * h_tilde


class AMPUGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        H = hidden_size
        self.x_to_main = nn.Linear(input_size, 4 * H)
        self.x_to_mi = nn.Linear(input_size, H, bias=False)
        self.h_to_gates = nn.Linear(H, 3 * H, bias=False)
        self.rh_to_cand = nn.Linear(H, 2 * H, bias=False)
        self.mi_b = nn.Parameter(torch.zeros(H))
        self.pu = ProductUnitLinear(input_size + H + H, H)

    def forward(self, x, h_prev):
        xw_main = self.x_to_main(x)
        xw_z, xw_r, xw_g, xw_add = xw_main.chunk(4, dim=-1)
        xw_mi = self.x_to_mi(x)
        hu_gates = self.h_to_gates(h_prev)
        hu_z, hu_r, hu_g = hu_gates.chunk(3, dim=-1)
        z = torch.sigmoid(xw_z + hu_z)
        r = torch.sigmoid(xw_r + hu_r)
        g = torch.sigmoid(xw_g + hu_g)
        r_h_prev = r * h_prev
        rhu_cand = self.rh_to_cand(r_h_prev)
        rhu_add, rhu_mi = rhu_cand.chunk(2, dim=-1)
        h_add = torch.tanh(xw_add + rhu_add)
        mi_pos = torch.exp(xw_mi * rhu_mi + self.mi_b)
        h_mul = self.pu(torch.cat([x, r_h_prev, mi_pos], dim=-1))
        h_tilde = g * h_mul + (1 - g) * h_add
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next
