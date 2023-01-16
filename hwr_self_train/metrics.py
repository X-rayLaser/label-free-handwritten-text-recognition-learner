import torch


class MetricsSetCalculator:
    def __init__(self, metrics, interval):
        """
        :param metrics: {'name': ('graph_leaf', Metric())}
        """
        self.metrics = metrics
        self.interval = interval
        self.running_metrics = {name: MovingAverage() for name in metrics}

    def __call__(self, iteration_log):
        if iteration_log.iteration % self.interval == 0:
            self.reset()

        with torch.no_grad():
            update_running_metrics(self.running_metrics, self.metrics, iteration_log.outputs)

        return self.running_metrics

    def reset(self):
        for metric_avg in self.running_metrics.values():
            metric_avg.reset()


class Metric:
    def __init__(self, name, metric_fn, metric_args, transform_fn):
        pass

    def __call__(self, **batch):
        pass


class MovingAverage:
    def __init__(self):
        self.x = 0
        self.num_updates = 0

    def reset(self):
        self.x = 0
        self.num_updates = 0

    def update(self, x):
        self.x += x
        self.num_updates += 1

    @property
    def value(self):
        return self.x / self.num_updates


def update_running_metrics(moving_averages, metrics, results_batch):
    for name, (leaf_name, metric_fn) in metrics.items():
        moving_averages[name].update(
            metric_fn(results_batch[leaf_name])
        )
