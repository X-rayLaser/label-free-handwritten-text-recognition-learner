from hwr_self_train.evaluation import evaluate
from hwr_self_train.training import print_metrics
from hwr_self_train.environment import Environment


if __name__ == '__main__':
    env = Environment()
    training_loop = env.training_loop

    for i in training_loop:
        epoch = env.epochs_trained + i + 1
        metrics = {}
        for task in env.eval_tasks:
            metrics.update(evaluate(task))

        print_metrics(metrics, epoch)
        env.history_saver.add_entry(epoch, metrics)
        env.save_checkpoint(epoch, metrics)

    print("Done")
