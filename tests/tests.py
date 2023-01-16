from pretrain import TrainingLoop

loop = TrainingLoop()

epoch_log = next(iter(loop))

assert epoch_log.epoch == 1
assert isinstance(epoch_log.metrics, dict)
assert isinstance(epoch_log.elapsed, float)
