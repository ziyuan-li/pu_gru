import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pu_gru import make_model, real_parameter_count
from .common import PROTOCOL, PROTOCOL_PATH, job_id, read_json, sha256, training_settings, write_json


def save_state(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def optimizer_for(model, settings):
    decay = settings['matrix_weight_decay']
    if decay == 0:
        return torch.optim.RAdam(model.parameters(), lr=settings['learning_rate'], weight_decay=0.0)
    matrix, other = [], []
    for name, parameter in model.named_parameters():
        (matrix if name.endswith('weight') and parameter.ndim > 1 else other).append(parameter)
    return torch.optim.RAdam([dict(params=matrix, weight_decay=decay),
                             dict(params=other, weight_decay=0.0)], lr=settings['learning_rate'])


def centralize(model):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if name.endswith('weight') and gradient is not None and gradient.ndim > 1:
                gradient.sub_(gradient.mean(dim=tuple(range(1, gradient.ndim)), keepdim=True))


def validate_bundle(folder, job):
    metadata = read_json(folder / 'metadata.json')
    spec = PROTOCOL['tasks'][job['task']]
    if metadata['task'] != job['task'] or metadata['split_seed'] != job['split_seed']:
        raise ValueError('Data bundle identity mismatch')
    for key in ('split_fractions', 'split_roles', 'selection', 'prior', 'target_unit', 'output_factor'):
        if metadata[key] != spec[key]:
            raise ValueError(f'Data bundle protocol mismatch: {key}')
    for name, digest in metadata['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError(f'Data bundle hash mismatch: {name}')
    return metadata


def load_tensors(folder, split, device):
    with np.load(folder / f'{split}.npz', allow_pickle=False) as data:
        x = torch.tensor(data['X'], dtype=torch.float32, device=device)
        y = torch.tensor(data['y'], dtype=torch.float32, device=device)
    return x, y


def predict(model, inputs):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(batch) for batch in inputs.split(256)])


def evaluate(model, folder, split, output, device, task):
    x, _ = load_tensors(folder, split, device)
    scaled = predict(model, x).cpu().numpy().astype(np.float64)
    scaler = read_json(folder / 'scalers.json')
    factor = PROTOCOL['tasks'][task]['output_factor']
    prediction = (scaled * np.asarray(scaler['y_scale']) + np.asarray(scaler['y_mean'])).ravel() * factor
    with np.load(folder / f'{split}.npz', allow_pickle=False) as data:
        true, prior = data['y_raw'].ravel() * factor, data['prior'].ravel() * factor
        if not np.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation predictions')
        np.savez_compressed(output / f'predictions_{split}.npz', ZN=data['ZN'],
                            true=true, prediction=prediction, prior=prior, context_ZN=data['context_ZN'])
    error = prediction - true
    return dict(samples=len(error), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(np.abs(error))), bias=float(np.mean(error)),
                prior_rmse=float(np.sqrt(np.mean((prior - true) ** 2))))


def batches(x, y, generator, loader, batch_size):
    if loader is not None:
        for a, b in loader:
            yield a.to(x.device), b.to(x.device)
    else:
        order = torch.randperm(len(x), generator=generator).to(x.device)
        for indices in order.split(batch_size):
            yield x[indices], y[indices]


def train(job, data_root, output_root, device_name='cpu', smoke_epochs=0):
    if job['training_seed'] < 0 or job['split_seed'] < 0 or not 0 <= smoke_epochs <= 10:
        raise ValueError('Invalid seed or smoke epoch count')
    folder = Path(data_root) / job['task'] / f"split_{job['split_seed']}"
    metadata = validate_bundle(folder, job)
    settings = training_settings(job['task'], job['model'])
    epochs = smoke_epochs or settings['epochs']
    output = Path(output_root) / ('smoke' if smoke_epochs else 'runs') / job_id(job)
    identity = dict(job=job, settings=settings, smoke_epochs=smoke_epochs,
                    data_sha256=sha256(folder / 'metadata.json'), protocol_sha256=sha256(PROTOCOL_PATH),
                    implementation_sha256={path.relative_to(PROTOCOL_PATH.parent.parent).as_posix(): sha256(path)
                        for package in ('pu_gru', 'experiments')
                        for path in sorted((PROTOCOL_PATH.parent.parent / package).glob('*.py'))})
    if (output / 'result.json').exists():
        previous = read_json(output / 'result.json')
        if previous['identity'] != identity or previous['status'] != ('smoke' if smoke_epochs else 'completed'):
            raise RuntimeError('Existing run is incomplete or has a different configuration')
        for name, digest in previous['artifact_hashes'].items():
            if sha256(output / name) != digest:
                raise ValueError(f'Existing artifact hash mismatch: {name}')
        return previous
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    seed_training(job['training_seed'])
    model = make_model(job['model'], job['task']).to(device)
    optimizer = optimizer_for(model, settings)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings['scheduler']['step_size'], gamma=settings['scheduler']['gamma'])
    generator = torch.Generator().manual_seed(job['training_seed'])
    x, y = load_tensors(folder, 'train', device)
    loader = None
    if job['group'] != 'main_mass':
        loader = DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=settings['batch_size'],
                            shuffle=True, generator=generator)
    has_validation = settings['selection'] == 'minimum_validation_mse'
    if has_validation:
        x_val, y_val = load_tensors(folder, 'val', device)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists() and read_json(config_path)['identity'] != identity:
        raise ValueError('Run configuration changed')
    write_json(config_path, dict(identity=identity, parameters=real_parameter_count(model),
                                 torch=torch.__version__, numpy=np.__version__, device=str(device)))
    write_json(output / 'scalers.json', read_json(folder / 'scalers.json'))
    best_loss, best_epoch, best_state = float('inf'), None, None
    history, completed = [], 0
    resume_path = output / 'resume.pth'
    if resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu', weights_only=True)
        if resume['identity'] != identity:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        generator.set_state(resume['generator'])
        completed, history = resume['completed_epochs'], resume['history']
        best_loss, best_epoch, best_state = resume['best_loss'], resume['best_epoch'], resume['best_state']
        torch.set_rng_state(resume['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
    start = time.monotonic()
    try:
        for epoch in range(completed + 1, epochs + 1):
            for name, parameter in model.named_parameters():
                if name.endswith('raw_theta'):
                    parameter.requires_grad_(epoch > settings['theta_frozen_epochs'])
            model.train()
            total = 0.0
            for batch_x, batch_y in batches(x, y, generator, loader, settings['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(batch_x), batch_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
                loss.backward()
                if settings['gradient_centralization']:
                    centralize(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(batch_x)
            record = dict(epoch=epoch, train_mse=total / len(x))
            if has_validation:
                val_loss = float(torch.nn.functional.mse_loss(predict(model, x_val), y_val))
                if not np.isfinite(val_loss):
                    raise FloatingPointError('Nonfinite validation loss')
                record['val_mse'] = val_loss
                if val_loss < best_loss:
                    best_loss, best_epoch, best_state = val_loss, epoch, cpu_state(model)
            scheduler.step()
            history.append(record)
            completed = epoch
            if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
                save_state(resume_path, dict(identity=identity, model=cpu_state(model), optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], completed_epochs=completed,
                           best_loss=best_loss, best_epoch=best_epoch, best_state=best_state, history=history))
                write_json(output / 'history.json', history)
                print(f"{job_id(job)} epoch={epoch}/{epochs} train_mse={record['train_mse']:.8g}", flush=True)
        save_state(output / 'final.pth', cpu_state(model))
        selected_epoch = best_epoch if has_validation else completed
        if has_validation:
            model.load_state_dict(best_state)
        save_state(output / 'selected.pth', cpu_state(model))
        roles = [role for role in metadata['split_roles'] if not smoke_epochs or role in ('train', 'val')]
        metrics = {role: evaluate(model, folder, role, output, device, job['task']) for role in roles}
        files = ['selected.pth', 'final.pth', 'config.json', 'scalers.json', 'history.json']
        files.extend(f'predictions_{role}.npz' for role in roles)
        result = dict(identity=identity, status='smoke' if smoke_epochs else 'completed', completed_epochs=completed,
                      selected_epoch=selected_epoch, selection=settings['selection'], metrics=metrics,
                      metric_unit=metadata['metric_unit'], duration_seconds=time.monotonic() - start,
                      artifact_hashes={name: sha256(output / name) for name in files})
        write_json(output / 'result.json', result)
        return result
    except Exception as exc:
        save_state(output / 'failed_state.pth', cpu_state(model))
        write_json(output / 'result.json', dict(identity=identity, status='failed', completed_epochs=completed,
                                               error=f'{type(exc).__name__}: {exc}'))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=PROTOCOL['tasks'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--split-seed', type=int, required=True)
    parser.add_argument('--training-seed', type=int, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke-epochs', type=int, default=0)
    args = parser.parse_args()
    group = 'radius' if args.task == 'radius' else ('main_mass' if args.model in PROTOCOL['main_mass']['models'] else 'additional_mass')
    job = dict(group=group, task=args.task, model=args.model, split_seed=args.split_seed, training_seed=args.training_seed)
    train(job, args.data_root, args.output_root, args.device, args.smoke_epochs)


if __name__ == '__main__':
    main()
