import torch

from pu_gru import make_matched_pair, make_model, real_parameter_count


def main():
    torch.manual_seed(0)
    model = make_model('cv_mi_pu_gru', task='radius')
    inputs = torch.randn(8, 5, 3)
    targets = torch.randn(8, 1)
    predictions = model(inputs)
    loss = (predictions - targets).square().mean()
    loss.backward()
    print('Prediction shape:', tuple(predictions.shape))
    print('Real parameter components:', real_parameter_count(model))
    print('Synthetic MSE:', loss.item())
    full, control = make_matched_pair('am', task='radius', activation='tanh')
    print('AM PU/Tanh components:', real_parameter_count(full), real_parameter_count(control))


if __name__ == '__main__':
    main()
