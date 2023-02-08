import torch

from collections import namedtuple
from .metrics import MovingAverage, update_running_metrics
from .formatters import show_progress_bar


def evaluate(task, supress_errors=True):
    """Compute metrics.

    :param task: instance of EvaluationTask
    :param supress_errors: When set to True (by default), it will silently catch exceptions
    such as torch.cuda.OutOfMemoryError. Otherwise, exceptions will propagate to the caller.
    :return: a dictionary mapping names of metrics to their computed values
    """
    recognizer = task.recognizer
    recognizer.neural_pipeline.eval_mode()

    metrics = task.metric_functions
    data_generator = task.data_loader

    num_batches = calculate_num_batches(task.num_batches, len(data_generator))

    moving_averages = {name: MovingAverage() for name in metrics}

    gen = show_progress_bar(data_generator, desc='Evaluating metrics: ',
                            num_iters=num_batches, cols=50)
    with torch.no_grad():
        for i, (images, transcripts) in enumerate(gen):
            if i >= num_batches:
                break

            args = [images] if task.close_loop_prediction else [images, transcripts]

            try:
                y_hat = recognizer(*args)
                d = dict(y=transcripts, y_hat=y_hat)
                update_running_metrics(moving_averages, metrics, d)
            except torch.cuda.OutOfMemoryError:
                if not supress_errors:
                    raise

    return {metric_name: avg.value for metric_name, avg in moving_averages.items()}


def calculate_num_batches(num_batches, batches_total):
    if isinstance(num_batches, float):
        fraction = num_batches
        num_batches = int(round(batches_total * fraction))
        num_batches = max(1, num_batches)
    return num_batches


EvaluationTask = namedtuple('EvaluationTask',
                            ['recognizer', 'data_loader', 'metric_functions',
                             'num_batches', 'close_loop_prediction'])
