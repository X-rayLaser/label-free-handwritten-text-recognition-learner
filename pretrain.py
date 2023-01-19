from hwr_self_train.evaluation import evaluate
from hwr_self_train.training import print_metrics
from hwr_self_train.checkpoints import (
    save_checkpoint,
    make_new_checkpoint
)
from configuration import Configuration
from hwr_self_train.environment import Environment


if __name__ == '__main__':

    env = Environment()

    for epoch in env.training_loop:
        metrics = {}
        for task in env.eval_tasks:
            metrics.update(evaluate(task))

        print_metrics(metrics, epoch)
        env.history_saver.add_entry(epoch, metrics)

        save_dir = make_new_checkpoint(Configuration.checkpoints_save_dir)
        save_checkpoint(env.neural_pipeline, save_dir, Configuration.device)
    print("Done")
