import json
import os
import shutil

import torch


def clean_metrics(metrics):
    res = {}
    for name, value in metrics.items():
        if hasattr(value, 'item'):
            value = value.item()
        res[name] = value

    return res


class CheckpointKeeper:
    def __init__(self, checkpoints_dir):
        self.checkpoints_dir = checkpoints_dir

    def get_latest_meta_data(self):
        highest = self._get_highest_checkpoint_number()
        checkpoint_folder = os.path.join(self.checkpoints_dir, str(highest))
        return Checkpoint(checkpoint_folder).meta_data

    def load_latest_checkpoint(self, trainable, device):
        highest = self._get_highest_checkpoint_number()
        highest_dir = os.path.join(self.checkpoints_dir, str(highest))
        return self.load_checkpoint(trainable, highest_dir, device)

    def load_checkpoint(self, trainable, checkpoint_dir, device):
        checkpoint = Checkpoint(checkpoint_dir)

        session_state = checkpoint.get_session_state(device)

        trainable.encoder.load_state_dict(session_state.encoder)
        trainable.encoder.to(device)

        trainable.decoder.load_state_dict(session_state.decoder)
        trainable.decoder.to(device)

        trainable.encoder_optimizer.load_state_dict(session_state.encoder_optimizer)
        trainable.decoder_optimizer.load_state_dict(session_state.decoder_optimizer)

    def make_new_checkpoint(self, trainable, device, epoch, metrics):
        try:
            highest = self._get_highest_checkpoint_number()
            checkpoint_name = str(highest + 1)
        except CheckpointsNotFound:
            checkpoint_name = '0'

        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        os.makedirs(checkpoint_path)
        Checkpoint.create(checkpoint_path, trainable, device, epoch, metrics)

    def _get_highest_checkpoint_number(self):
        checkpoints = []
        for folder in os.listdir(self.checkpoints_dir):
            try:
                checkpoints.append(int(folder))
            except ValueError:
                pass

        if not checkpoints:
            raise CheckpointsNotFound()

        return max(checkpoints)

    def get_latest_checkpoint_dir(self):
        highest = self._get_highest_checkpoint_number()
        checkpoint_name = str(highest)
        return os.path.join(self.checkpoints_dir, checkpoint_name)


class Checkpoint:
    def __init__(self, folder):
        self._folder = folder
        meta_path = self._metadata_path(self._folder)

        with open(meta_path, encoding='utf-8') as f:
            self._meta_data = json.loads(f.read())

    @classmethod
    def create(cls, save_dir, trainable, device, epoch, metrics):
        checkpoint_path = cls._state_path(save_dir)
        meta_path = cls._metadata_path(save_dir)

        with open(meta_path, 'w', encoding='utf-8') as f:
            metrics = clean_metrics(metrics)
            meta_data = {
                'device': str(device),
                'epoch': epoch,
                'metrics': metrics
            }
            f.write(json.dumps(meta_data))

        models_dict = dict(encoder=trainable.encoder.state_dict(), decoder=trainable.decoder.state_dict())

        optimizers_dict = {
            'encoder_optimizer': trainable.encoder_optimizer.state_dict(),
            'decoder_optimizer': trainable.decoder_optimizer.state_dict()
        }

        torch.save({
            'models': models_dict,
            'optimizers': optimizers_dict
        }, checkpoint_path)
        return cls(save_dir)

    @classmethod
    def _state_path(cls, folder):
        return os.path.join(folder, 'checkpoint.pt')

    @classmethod
    def _metadata_path(cls, folder):
        return os.path.join(folder, 'metadata.txt')

    def get_session_state(self, device):
        state_path = self._state_path(self._folder)
        if self.meta_data["device"] == device:
            state_dict = torch.load(state_path)
        else:
            state_dict = torch.load(state_path, map_location=device)
        return SessionState(state_dict)

    @property
    def meta_data(self):
        return self._meta_data


class SessionState:
    def __init__(self, state_dict):
        self.state_dict = state_dict

    @property
    def encoder(self):
        return self.state_dict["models"]["encoder"]

    @property
    def decoder(self):
        return self.state_dict["models"]["decoder"]

    @property
    def encoder_optimizer(self):
        return self.state_dict["optimizers"]["encoder_optimizer"]

    @property
    def decoder_optimizer(self):
        return self.state_dict["optimizers"]["decoder_optimizer"]


class SessionDirectoryLayout:
    def __init__(self, session_dir):
        self.session = session_dir

        self.checkpoints = os.path.join(session_dir, "checkpoints")
        self.tuning_checkpoints = os.path.join(session_dir, "tuning_checkpoints")

        self.history = os.path.join(session_dir, "history.csv")
        self.tuning_history = os.path.join(session_dir, "tuning_history.csv")

        self.model_spec = os.path.join(session_dir, "model_spec.json")

    def make_session_dir(self):
        """Creates session directory if it does not exist"""
        if not os.path.exists(self.session):
            os.makedirs(self.session)

    def make_checkpoints_dir(self):
        """Creates checkpoints subdirectory if it does not exist"""
        if not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)

    def remove_history(self):
        self._remove_history_file(self.history)

    def remove_tuning_history(self):
        self._remove_history_file(self.tuning_history)

    def tuning_checkpoint_exists(self):
        path = os.path.join(self.tuning_checkpoints, '0')
        return os.path.exists(path)

    def create_tuning_checkpoint(self):
        """Copy checkpoint of a pretrained model into a directory for tuning checkpoints
        (only if this hasn't been already done)
        """
        if self.tuning_checkpoint_exists():
            return

        keeper = CheckpointKeeper(self.checkpoints)
        checkpoint_path = keeper.get_latest_checkpoint_dir()
        dest_path = os.path.join(self.tuning_checkpoints, '0')
        shutil.copytree(checkpoint_path, dest_path)

        self.update_meta_data(dest_path)

        self.remove_tuning_history()

    def update_meta_data(self, checkpoint_dir):
        tuning_keeper = CheckpointKeeper(self.tuning_checkpoints)
        meta_data = tuning_keeper.get_latest_meta_data()
        meta_data["epoch"] = 0

        # todo: change extension to json
        meta_path = os.path.join(checkpoint_dir, "metadata.txt")
        meta_json = json.dumps(meta_data)
        with open(meta_path, "w") as f:
            f.write(meta_json)

    def _remove_history_file(self, history_path):
        if os.path.isfile(history_path):
            os.remove(history_path)


class CheckpointsNotFound(Exception):
    """Raised when trying to load a checkpoint from a folder containing none of them"""
