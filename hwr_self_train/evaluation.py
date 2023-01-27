import torch

from collections import namedtuple
from .metrics import MovingAverage, update_running_metrics
from .formatters import ProgressBar


def evaluate(task, supress_errors=True):
    """Compute metrics.

    :param task: instance of EvaluationTask
    :param num_batches: Number of batches used to compute metrics on.
    Integer value is interpreted as precise number batches.
    Floating point number in 0-1 range is interpreted as a fraction of all batches in the pipeline.
    :param supress_errors: When set to True (by default), it will silently catch exceptions
    such as torch.cuda.OutOfMemoryError. Otherwise, exceptions will propagate to the caller.
    :return: a dictionary mapping names of metrics to their computed values
    """
    recognizer = task.recognizer
    metrics = task.metric_functions

    recognizer.neural_pipeline.eval_mode()

    data_generator = task.data_loader

    num_batches = task.num_batches

    if isinstance(num_batches, float):
        fraction = num_batches
        batches_total = len(data_generator)
        num_batches = int(round(batches_total * fraction))
        num_batches = max(1, num_batches)

    moving_averages = {name: MovingAverage() for name in metrics}

    whitespaces = ' ' * 150
    print(f'\r{whitespaces}', end='')

    progress_bar = ProgressBar()

    with torch.no_grad():
        for i, (images, transcripts) in enumerate(data_generator):
            if i >= num_batches:
                break

            try:
                if task.close_loop_prediction:
                    y_hat = recognizer(images)
                else:
                    y_hat = recognizer(images, transcripts)
            except torch.cuda.OutOfMemoryError:
                if not supress_errors:
                    raise
                continue

            d = dict(y=transcripts, y_hat=y_hat)
            update_running_metrics(moving_averages, metrics, d)

            step_number = i + 1
            progress = progress_bar.updated(step_number, num_batches, cols=50)
            print(f'\rEvaluating metrics: {progress} {step_number}/{num_batches}', end='')

    return {metric_name: avg.value for metric_name, avg in moving_averages.items()}


EvaluationTask = namedtuple('EvaluationTask',
                            ['recognizer', 'data_loader', 'metric_functions',
                             'num_batches', 'close_loop_prediction'])
