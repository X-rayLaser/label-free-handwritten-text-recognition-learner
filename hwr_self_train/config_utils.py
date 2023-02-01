from hwr_self_train.metrics import Metric


def create_metric(name, metric_fn, transform_fn):
    return Metric(
        name, metric_fn=metric_fn, metric_args=["y_hat", "y"], transform_fn=transform_fn
    )


def create_optimizer(model, optimizer_conf):
    optimizer_class = optimizer_conf['class']
    kwargs = optimizer_conf['kwargs']
    return optimizer_class(model.parameters(), **kwargs)


def prepare_loss(loss_conf):
    loss_class = loss_conf["class"]
    loss_kwargs = loss_conf["kwargs"]
    loss_transform = loss_conf["transform"]
    loss_function = loss_class(**loss_kwargs)
    return create_metric('loss', loss_function, loss_transform)


def prepare_metrics(metrics_conf):
    metric_fns = {}
    for name, spec in metrics_conf.items():
        metric_class = spec['class']
        metric_args = spec.get('args', [])
        metric_kwargs = spec.get('kwargs', {})
        transform_fn = spec['transform']
        metric_fn = metric_class(*metric_args, **metric_kwargs)

        metric_fns[name] = create_metric(name, metric_fn, transform_fn)

    return metric_fns
