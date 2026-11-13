import hashlib
import json
from pathlib import Path

from pu_gru import available_models
from pu_gru.models import PHASE_CONVENTIONS


PROTOCOL_PATH = Path(__file__).with_name('protocol.json')
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
GROUPS = ('main_mass', 'additional_mass', 'radius')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_id(job):
    return f"{job['group']}/{job['task']}/split_{job['split_seed']}/{job['model']}/seed_{job['training_seed']}"


def plan(group):
    spec = PROTOCOL[group]
    return [dict(group=group, task=task, split_seed=split, model=model, training_seed=seed)
            for task in spec['tasks'] for split in spec['split_seeds']
            for model in spec['models'] for seed in spec['training_seeds']]


def table_plan(number):
    if number == 1:
        return plan('main_mass')
    if number == 2:
        return plan('radius')
    if number == 3:
        references = [job for job in plan('main_mass')
                      if job['model'] in ('cv_mi_pu_gru', 'cv_am_pu_gru', 'cv_gru')]
        return references + plan('additional_mass')
    raise ValueError('Table must be 1, 2, or 3')


def training_settings(task, model):
    if model not in available_models(task):
        raise ValueError(f'Unsupported model/task: {model}/{task}')
    radius = task == 'radius'
    return dict(epochs=PROTOCOL['epochs'], batch_size=PROTOCOL['batch_size'],
                learning_rate=PROTOCOL['learning_rate'], optimizer=PROTOCOL['optimizer'],
                scheduler=PROTOCOL['scheduler'], gradient_clip_norm=PROTOCOL['gradient_clip_norm'],
                matrix_weight_decay=1e-4 if radius else 0.0,
                theta_frozen_epochs=5 if radius else 0,
                negative_axis_phase=PHASE_CONVENTIONS[task],
                gradient_centralization=not radius and model in PROTOCOL['main_mass']['models'] and model != 'cv_gru',
                selection=PROTOCOL['tasks'][task]['selection'])
