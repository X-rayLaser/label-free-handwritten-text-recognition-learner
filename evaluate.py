import argparse
from hwr_self_train.environment import Environment
from hwr_self_train.evaluation import evaluate
from hwr_self_train.formatters import Formatter
from hwr_self_train.checkpoints import SessionDirectoryLayout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a model'
    )

    parser.add_argument('session_dir', type=str,
                        help='Path to the location of session directory')
    args = parser.parse_args()
    config = SessionDirectoryLayout(args.session_dir).load_config()
    env = Environment(config)

    metrics = {}
    for task in env.eval_tasks:
        metrics.update(evaluate(task))

    formatter = Formatter()

    final_metrics_string = formatter.format_metrics(metrics)

    whitespaces = ' ' * 150
    print(f'\r{whitespaces}\r{final_metrics_string}')
