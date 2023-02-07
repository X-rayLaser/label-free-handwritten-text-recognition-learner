import argparse
from hwr_self_train.evaluation import evaluate
from hwr_self_train.training import print_metrics
from hwr_self_train.environment import Environment
from hwr_self_train.checkpoints import SessionDirectoryLayout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('session_dir', type=str, default='',
                        help='Location of the session directory')
    args = parser.parse_args()

    config = SessionDirectoryLayout(args.session_dir).load_config()
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
