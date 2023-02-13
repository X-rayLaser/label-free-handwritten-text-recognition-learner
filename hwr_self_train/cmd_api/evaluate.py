from hwr_self_train.environment import Environment
from hwr_self_train.evaluation import evaluate
from hwr_self_train.formatters import Formatter
from hwr_self_train.session import SessionDirectoryLayout
from .base import Command


class EvaluateCommand(Command):
    name = 'evaluate'
    help = 'Evaluate a model using a dataset and metrics specified in configuration file'

    def configure_parser(self, parser):
        parser.add_argument('session_dir', type=str,
                            help='Path to the location of session directory')

    def __call__(self, args):
        config = SessionDirectoryLayout(args.session_dir).load_config()
        env = Environment(config)

        metrics = {}
        for task in env.eval_tasks:
            metrics.update(evaluate(task))

        formatter = Formatter()

        final_metrics_string = formatter.format_metrics(metrics)

        whitespaces = ' ' * 150
        print(f'\r{whitespaces}\r{final_metrics_string}')
