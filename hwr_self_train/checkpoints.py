import os
import torch
from torch.optim import Adam

from .models import build_networks
from .training import TrainableEncoderDecoder


def save_checkpoint(trainable, save_dir, device):
    os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
    device_path = os.path.join(save_dir, 'device.txt')

    with open(device_path, 'w', encoding='utf-8') as f:
        f.write(str(device))

    models_dict = dict(encoder=trainable.encoder.state_dict(), decoder=trainable.decoder.state_dict)

    optimizers_dict = {
        'encoder_optimizer': trainable.encoder_optimizer.state_dict(),
        'decoder_optimizer': trainable.decoder_optimizer.state_dict()
    }

    torch.save({
        'models': models_dict,
        'optimizers': optimizers_dict
    }, checkpoint_path)


def load_checkpoint(checkpoint_dir, device):
    encoder, decoder = build_networks()

    encoder_optimizer = Adam(encoder.parameters(), lr=0.0001)
    decoder_optimizer = Adam(decoder.parameters(), 0.0001)

    state_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    device_path = os.path.join(checkpoint_dir, 'device.txt')

    with open(device_path, encoding='utf-8') as f:
        checkpoint_device = torch.device(f.read())

    if checkpoint_device == device:
        checkpoint = torch.load(state_path)
    else:
        checkpoint = torch.load(state_path, map_location=device)

    encoder.load_state_dict(checkpoint['models']['encoder'])
    encoder.to(device)

    decoder.load_state_dict(checkpoint['models']['decoder'])
    decoder.to(device)

    encoder_optimizer.load_state_dict(checkpoint['optimizers']['encoder_optimizer'])
    decoder_optimizer.load_state_dict(checkpoint['optimizers']['decoder_optimizer'])

    trainable = TrainableEncoderDecoder(encoder, decoder, encoder_optimizer, decoder_optimizer)
    return trainable
