import argparse
import os
from hwr_self_train.evaluation import evaluate
from hwr_self_train.training import print_metrics
from hwr_self_train.environment import Environment
from hwr_self_train.config_utils import load_conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('session_dir', type=str, default='',
                        help='Location of the session directory')
    args = parser.parse_args()

    config_path = os.path.join(args.session_dir, "config.json")
    with open(config_path) as f:
        json_str = f.read()

    config = load_conf(json_str)
    env = Environment(config)
    training_loop = env.training_loop

    for epoch in training_loop:
        metrics = {}
        for task in env.eval_tasks:
            metrics.update(evaluate(task))

        print_metrics(metrics, epoch)
        env.history_saver.add_entry(epoch, metrics)
        env.save_checkpoint(epoch, metrics)

    print("Done")
