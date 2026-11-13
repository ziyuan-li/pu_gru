import argparse
from pathlib import Path

import numpy as np
import torch

from pu_gru.checkpoints import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, args.device)
    inputs = torch.as_tensor(np.load(args.inputs, allow_pickle=False), dtype=torch.float64)
    with torch.inference_mode():
        predictions = torch.cat([model(batch) for batch in inputs.split(256)]).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, predictions)


if __name__ == '__main__':
    main()
