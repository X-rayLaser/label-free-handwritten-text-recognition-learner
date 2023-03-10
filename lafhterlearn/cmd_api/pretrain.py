from lafhterlearn.evaluation import evaluate
from lafhterlearn.training import print_metrics
from lafhterlearn.environment import Environment
from lafhterlearn.session import SessionDirectoryLayout

from .base import Command


class PretrainCommand(Command):
    name = 'pretrain'
    help = 'Start/resume pretraining of the model on synthetic data'

    def configure_parser(self, parser):
        configure_parser(parser)

    def __call__(self, args):
        run(args)


def configure_parser(parser):
    parser.add_argument('session_dir', type=str, default='',
                        help='Location of the session directory')


def run(args):
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
