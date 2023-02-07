from hwr_self_train.metrics import Metric
from hwr_self_train.utils import instantiate_class


def create_metric(name, metric_fn, transform_fn):
    return Metric(
        name, metric_fn=metric_fn, metric_args=["y_hat", "y"], transform_fn=transform_fn
    )


def create_optimizer(model, optimizer_conf):
    optimizer_name = optimizer_conf['class']
    kwargs = optimizer_conf['kwargs']
    return instantiate_class(optimizer_name, model.parameters(), **kwargs)


def prepare_loss(loss_conf):
    loss_class = loss_conf["class"]
    loss_kwargs = loss_conf["kwargs"]
    loss_function = instantiate_class(loss_class, **loss_kwargs)

    loss_transform_conf = loss_conf["transform"]
    transform_class = loss_transform_conf["class"]
    transform_kwargs = loss_transform_conf["kwargs"]
    loss_transform = instantiate_class(transform_class, **transform_kwargs)

    return create_metric('loss', loss_function, loss_transform)


def prepare_metrics(metrics_conf):
    metric_fns = {}
    for name, spec in metrics_conf.items():
        metric_class = spec['class']
        metric_args = spec.get('args', [])
        metric_kwargs = spec.get('kwargs', {})
        metric_fn = instantiate_class(metric_class, *metric_args, **metric_kwargs)

        transform_conf = spec['transform']
        transform_class = transform_conf['class']
        transform_kwargs = transform_conf['kwargs']
        transform_fn = instantiate_class(transform_class, **transform_kwargs)
        metric_fns[name] = create_metric(name, metric_fn, transform_fn)

    return metric_fns
