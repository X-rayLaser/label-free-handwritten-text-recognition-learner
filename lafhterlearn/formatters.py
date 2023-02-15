class Formatter:
    def __init__(self):
        self.progress_bar = ProgressBar()

    def format_epoch(self, epoch):
        return f'Epoch {epoch:5}'

    def format_metrics(self, metrics):
        metric_strings = [f'{name} {value:6.4f}' for name, value in metrics.items()]
        s = ', '.join(metric_strings)
        return f'[{s}]'

    def __call__(self, epoch, iteration, num_iterations, running_metrics):
        computed_metrics = {name: avg.value for name, avg in running_metrics.items()}

        metrics_str = self.format_metrics(computed_metrics)

        progress = self.progress_bar.updated(iteration, num_iterations)
        epoch_str = self.format_epoch(epoch)
        return f'{epoch_str} {metrics_str} {progress} {iteration} / {num_iterations}'


class ProgressBar:
    def updated(self, step, num_steps, fill='=', empty='.', cols=20):
        filled_chars = int(step / num_steps * cols)
        remaining_chars = cols - filled_chars
        return fill * filled_chars + '>' + empty * remaining_chars


def show_progress_bar(iterable, desc='', num_iters=None, cols=50):
    whitespaces = ' ' * 150
    print(f'\r{whitespaces}', end='')
    progress_bar = ProgressBar()

    if not num_iters and hasattr(iterable, '__len__'):
        num_iters = len(iterable)

    for i, data in enumerate(iterable):
        step_number = i + 1
        if num_iters:
            progress = progress_bar.updated(step_number, num_iters, cols=cols)
            msg = f'\r{desc}{progress} {step_number}/{num_iters}'
        else:
            max_val = 10000
            progress = progress_bar.updated(step_number % max_val, max_val, cols=cols)
            msg = f'\r{desc}{progress} Iterations {step_number}'
        print(msg, end='')
        yield data
