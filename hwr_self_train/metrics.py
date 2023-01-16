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
            d = dict(y=iteration_log.targets, y_hat=iteration_log.outputs)
            update_running_metrics(self.running_metrics, self.metrics, d)

        return self.running_metrics

    def reset(self):
        for metric_avg in self.running_metrics.values():
            metric_avg.reset()


class Metric:
    def __init__(self, name, metric_fn, metric_args, transform_fn=None):
        """

        :param name: A string that identifies the metric
        :param metric_fn: a callable that maps variable number of tensors to a scalar
        :param metric_args: tensor names of batch that are used to compute the metric on
        :type metric_args: iterable of type str
        :param transform_fn: a callable that transforms/reduces # of the metrics arguments
        """
        self.name = name
        self.metric_fn = metric_fn
        self.metric_args = metric_args
        self.transform_fn = transform_fn or identity

    def rename_and_clone(self, new_name):
        return self.__class__(new_name, self.metric_fn, self.metric_args, self.transform_fn)

    def __call__(self, **batch):
        """Given the batch of predictions and targets, compute the metric.

        The method will first extract all arguments that metric
        function expects from batch. Then it will apply transforms to them.
        Finally, it will place all tensors on a specified device and compute the metric value.

        Note that this method will also compute gradients with respect to tensors that participate in
        the metrics calculation. If this is not required, one should manually detach() tensors or
        make the call under no_grad() context manager.

        :param batch: a dictionary mapping a name of a metric argument to a corresponding tensor
        :type batch: Dict[str -> torch.tensor]
        :return: a metric scalar
        :rtype: degenerate tensor of shape ()
        """

        lookup_table = batch
        inputs = [lookup_table[arg] for arg in self.metric_args]

        device = self.fastest_device(inputs)

        inputs = self.change_device(inputs, device)
        inputs = self.transform_fn(*inputs)
        inputs = self.change_device(inputs, device)  # the above operation could change devices

        fn = self.metric_fn.to(device) if hasattr(self.metric_fn, 'to') else self.metric_fn
        return fn(*inputs)

    def fastest_device(self, inputs):
        """Returns cuda device if at least 1 input tensor is on cuda,
        otherwise return cpu device"""
        devices = [v.device for v in inputs if hasattr(v, 'device')]

        for d in devices:
            if 'cuda' in str(d):
                return d
        return torch.device('cpu')

    def change_device(self, tensors, device):
        """Moves all tensors that participate in metric calculation to a given device

        :param tensors: a list of tensors
        :param device: torch.device instance
        :return: a list of tensors
        """

        # some tensor may already be on the right device, if so they kept unchanged
        return [arg if not hasattr(arg, 'device') or arg.device == device else arg.to(device)
                for arg in tensors]


def identity(*inputs):
    return inputs


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
    for name, metric_fn in metrics.items():
        moving_averages[name].update(
            metric_fn(**results_batch)
        )
