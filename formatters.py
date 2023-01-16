class Formatter:
    def __init__(self):
        self.progress_bar = ProgressBar()

    def format_epoch(self, epoch):
        return f'Epoch {epoch:5}'

    def format_metrics(self, metrics, validation=False):
        # todo: seems method is never called with validation flag=True
        if validation:
            metrics = {f'val {k}': v for k, v in metrics.items()}

        metric_strings = [f'{name} {value:6.4f}' for name, value in metrics.items()]
        s = ', '.join(metric_strings)
        return f'[{s}]'

    def __call__(self, epoch, iteration, num_iterations, running_metrics):
        computed_metrics = {name: avg.value for name, avg in running_metrics.items()}

        metrics_str = self.format_metrics(computed_metrics)

        progress = self.progress_bar.updated(iteration, num_iterations)
        epoch_str = self.format_epoch(epoch)
        # todo: show also time elapsed
        return f'{epoch_str} {metrics_str} {progress} {iteration} / {num_iterations}'


class ProgressBar:
    def updated(self, step, num_steps, fill='=', empty='.', cols=20):
        filled_chars = int(step / num_steps * cols)
        remaining_chars = cols - filled_chars
        return fill * filled_chars + '>' + empty * remaining_chars
