import json
import os

import torch


def save_checkpoint(trainable, save_dir, device, epoch, metrics):
    os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
    meta_path = os.path.join(save_dir, 'meta.txt')

    with open(meta_path, 'w', encoding='utf-8') as f:
        meta_data = {
            'device': str(device),
            'epoch': epoch,
            'metrics': metrics
        }
        f.write(json.dumps(meta_data))

    models_dict = dict(encoder=trainable.encoder.state_dict(), decoder=trainable.decoder.state_dict)

    optimizers_dict = {
        'encoder_optimizer': trainable.encoder_optimizer.state_dict(),
        'decoder_optimizer': trainable.decoder_optimizer.state_dict()
    }

    torch.save({
        'models': models_dict,
        'optimizers': optimizers_dict
    }, checkpoint_path)


def get_checkpoint_meta(checkpoint_dir):
    meta_path = os.path.join(checkpoint_dir, 'meta.txt')
    with open(meta_path, encoding='utf-8') as f:
        return json.loads(f.read())


def get_latest_meta_data(checkpoints_dir):
    highest = get_highest_checkpoint_number(checkpoints_dir)
    meta_path = os.path.join(checkpoints_dir, str(highest))
    return get_checkpoint_meta(meta_path)


def load_checkpoint(trainable, checkpoint_dir, device):
    state_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    meta_path = os.path.join(checkpoint_dir, 'meta.txt')

    with open(meta_path, encoding='utf-8') as f:
        d = json.loads(f.read())
        checkpoint_device = torch.device(d["device"])

    if checkpoint_device == device:
        checkpoint = torch.load(state_path)
    else:
        checkpoint = torch.load(state_path, map_location=device)

    trainable.encoder.load_state_dict(checkpoint['models']['encoder'])
    trainable.encoder.to(device)

    trainable.decoder.load_state_dict(checkpoint['models']['decoder'])
    trainable.decoder.to(device)

    trainable.encoder_optimizer.load_state_dict(checkpoint['optimizers']['encoder_optimizer'])
    trainable.decoder_optimizer.load_state_dict(checkpoint['optimizers']['decoder_optimizer'])


def make_new_checkpoint(base_dir):
    try:
        highest = get_highest_checkpoint_number(base_dir)
    except CheckpointsNotFound:
        highest = 0

    checkpoint_name = str(highest + 1)
    checkpoint_path = os.path.join(base_dir, checkpoint_name)
    os.makedirs(checkpoint_path)
    return checkpoint_path


def load_latest_checkpoint(trainable, checkpoints_dir, device):
    highest = get_highest_checkpoint_number(checkpoints_dir)
    highest_dir = os.path.join(checkpoints_dir, str(highest))
    return load_checkpoint(trainable, highest_dir, device)


def get_highest_checkpoint_number(checkpoints_dir):
    checkpoints = []
    for folder in os.listdir(checkpoints_dir):
        try:
            checkpoints.append(int(folder))
        except ValueError:
            pass

    if not checkpoints:
        raise CheckpointsNotFound()

    return max(checkpoints)


class CheckpointsNotFound(Exception):
    """Raised when trying to load a checkpoint from a folder containing none of them"""
