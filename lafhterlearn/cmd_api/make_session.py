import os
import importlib
import json

import torch

from lafhterlearn.configuration import Configuration
from lafhterlearn.session import SessionDirectoryLayout, CheckpointKeeper
from lafhterlearn.models import build_networks_spec
from lafhterlearn.environment import create_neural_pipeline
from .base import Command


class CreateSessionCommand(Command):
    name = 'make_session'
    help = 'Create a fresh training session'

    def configure_parser(self, parser):
        configure_parser(parser)

    def __call__(self, args):
        run(args)


def configure_parser(parser):
    parser.add_argument('config_file', type=str,
                        help='Location of the configuration file (must be a Python module).')


def run(args):
    prepare_session(get_config(args.config_file))


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
