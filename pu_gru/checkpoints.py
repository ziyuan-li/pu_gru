import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .models import make_model, validate_sequence


class CheckpointPredictor(nn.Module):
    def __init__(self, model, scalers, metadata):
        super().__init__()
        self.model = model
        self.metadata = metadata
        for key in ('x_mean', 'x_scale', 'y_mean', 'y_scale'):
            self.register_buffer(key, torch.tensor(scalers[key], dtype=torch.float64))
        if self.x_mean.numel() != 15 or self.x_scale.numel() != 15 or self.y_mean.numel() != 1 or self.y_scale.numel() != 1:
            raise ValueError('Invalid scaler dimensions')
        if any(not torch.isfinite(value).all() for value in (self.x_mean, self.x_scale, self.y_mean, self.y_scale)):
            raise ValueError('Nonfinite scalers')
        if (self.x_scale <= 0).any() or (self.y_scale <= 0).any():
            raise ValueError('Nonpositive scaler scale')

    def forward(self, raw_inputs):
        validate_sequence(raw_inputs, 3)
        if not torch.isfinite(raw_inputs).all():
            raise ValueError('Nonfinite input sequences')
        inputs = raw_inputs.to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        scaled = ((inputs.flatten(start_dim=1) - self.x_mean) / self.x_scale).reshape(-1, 5, 3)
        prediction = self.model(scaled)
        return (prediction.to(torch.float64) * self.y_scale + self.y_mean) * self.metadata['output_factor']


def load_checkpoint(path, device='cpu'):
    path = Path(path).resolve()
    entry = None
    for parent in path.parents:
        manifest_path = parent / 'manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        matches = [item for item in manifest.get('checkpoints', []) if item['path'] == path.relative_to(parent).as_posix()]
        if len(matches) == 1:
            entry = matches[0]
            scaler_path = (parent / entry['scalers']).resolve()
            if not scaler_path.is_relative_to(parent):
                raise ValueError('Invalid scaler path')
            break
    if entry is None:
        result = json.loads((path.parent / 'result.json').read_text(encoding='utf-8'))
        if result['status'] != 'completed' or path.name != 'selected.pth':
            raise ValueError('Expected a completed selected checkpoint')
        job = result['identity']['job']
        entry = dict(**job, sha256=result['artifact_hashes'][path.name],
                     output_factor=1.0 if job['task'] == 'radius' else 0.001,
                     metric_unit=result['metric_unit'])
        scaler_path = path.parent / 'scalers.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    if 'scalers_sha256' in entry and hashlib.sha256(scaler_path.read_bytes()).hexdigest() != entry['scalers_sha256']:
        raise ValueError('Scaler hash mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Expected a tensor state dictionary')
    model = make_model(entry['model'], entry['task'])
    model.load_state_dict(state, strict=True)
    predictor = CheckpointPredictor(model, json.loads(scaler_path.read_text(encoding='utf-8')), entry)
    return predictor.to(device).eval()
