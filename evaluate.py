from hwr_self_train.environment import Environment
from hwr_self_train.evaluation import evaluate
from hwr_self_train.formatters import Formatter


if __name__ == '__main__':
    env = Environment()
    metrics = {}
    for task in env.eval_tasks:
        metrics.update(evaluate(task))

    formatter = Formatter()

    final_metrics_string = formatter.format_metrics(metrics)

    whitespaces = ' ' * 150
    print(f'\r{whitespaces}\r{final_metrics_string}')
