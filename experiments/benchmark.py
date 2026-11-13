import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn

from pu_gru import make_model, real_parameter_count
from pu_gru.complex import ProductUnitLinearC
from pu_gru.controls import ComplexAffineSplit
from pu_gru.real import ProductUnitLinear
from .common import write_json


MODELS = ('cv_mi_pu_gru', 'cv_am_pu_gru', 'real_gru', 'real_mi_pu_gru',
          'real_am_pu_gru', 'cv_gru', 'cv_mi_gru', 'cv_am_gru', 'cv_pu_gru', 'mlp')


def linear_flops(model, inputs):
    operations, handles = [], []
    def count(layer, args, output):
        rows, columns = layer.weight.shape
        vectors = args[0].numel() // columns
        mac_cost = 8 if layer.weight.is_complex() else 2
        bias_cost = 2 if layer.weight.is_complex() else 1
        operations.append(vectors * rows * columns * mac_cost)
        if layer.bias is not None:
            operations.append(vectors * rows * bias_cost)
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, ProductUnitLinear, ProductUnitLinearC, ComplexAffineSplit)):
            handles.append(layer.register_forward_hook(count))
    try:
        with torch.inference_mode():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return sum(operations) // len(inputs)


def latency(model, inputs, warmup=100, blocks=7, calls=200):
    if inputs.device.type != 'cuda':
        raise ValueError('GPU-resident inputs are required')
    if warmup < 0 or blocks < 2 or calls < 1:
        raise ValueError('Invalid timing configuration')
    values = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        for _ in range(blocks):
            torch.cuda.synchronize(inputs.device)
            start = time.perf_counter()
            for _ in range(calls):
                result = model(inputs)
            torch.cuda.synchronize(inputs.device)
            values.append((time.perf_counter() - start) * 1000 / calls)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite inference output')
    quartiles = statistics.quantiles(values, n=4, method='inclusive')
    return dict(median_ms_per_batch=statistics.median(values), q25_ms_per_batch=quartiles[0],
                q75_ms_per_batch=quartiles[2], block_means_ms=values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('CUDA is required')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    for name in args.models:
        torch.manual_seed(0)
        model = make_model(name, 'radius').to(device).eval()
        for batch in (1, 64):
            inputs = torch.randn(batch, 5, 3, device=device)
            row = dict(model=name, batch=batch, real_components=real_parameter_count(model),
                       linear_flops_per_sample=linear_flops(model, inputs), **latency(model, inputs))
            records.append(row)
            print(json.dumps(row), flush=True)
    write_json(args.output, dict(task='radius', torch=torch.__version__, cuda=torch.version.cuda,
                                gpu=torch.cuda.get_device_name(device), warmup=100, blocks=7,
                                calls_per_block=200, mixed_precision=False, tf32=False, results=records))


if __name__ == '__main__':
    main()
