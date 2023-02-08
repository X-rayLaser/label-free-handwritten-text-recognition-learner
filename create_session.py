import argparse
import os
import importlib
import json

import torch

from hwr_self_train.configuration import Configuration
from hwr_self_train.session import SessionDirectoryLayout, CheckpointKeeper
from hwr_self_train.models import build_networks_spec
from hwr_self_train.environment import create_neural_pipeline


def get_config(config_file) -> Configuration:
    conf_module = importlib.import_module(config_file)
    return conf_module.Configuration()


def prepare_session(config):
    if os.path.exists(config.session_dir):
        print(f'Session already exists in "{config.session_dir}"')
        return

    session_layout = SessionDirectoryLayout(config.session_dir)
    session_layout.make_session_dir()
    session_layout.make_checkpoints_dir()

    spec = build_networks_spec(charset=config.charset,
                               image_height=config.image_height,
                               hidden_size=config.hidden_size,
                               **config.decoder_params)

    device = torch.device(config.device)
    neural_pipeline = create_neural_pipeline(device, spec, config)

    keeper = CheckpointKeeper(session_layout.checkpoints)

    with open(session_layout.model_spec, 'w') as f:
        f.write(json.dumps(spec))

    neural_pipeline.encoder.to(neural_pipeline.device)
    neural_pipeline.decoder.to(neural_pipeline.device)
    keeper.make_new_checkpoint(neural_pipeline, device, 0, metrics={})

    json_str = config.to_json()

    config_save_path = os.path.join(config.session_dir, "config.json")
    with open(config_save_path, 'w') as f:
        f.write(json_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config_file', type=str, default='',
                        help='Location of the configuration file (must be a Python module).')
    args = parser.parse_args()
    prepare_session(get_config(args.config_file))
