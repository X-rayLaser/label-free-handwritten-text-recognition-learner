from unittest import TestCase
from lafhterlearn.metrics import Metric


class MetricTests(TestCase):
    def test_metric_without_arguments(self):
        def metric_fn():
            return 32

        metric = Metric("metric", metric_fn, [], None)
        self.assertEqual(32, metric())

    def test_metric_with_one_argument(self):
        def metric_fn(x): return x**2

        metric = Metric("metric", metric_fn, ["x"], None)
        self.assertEqual(9, metric(x=3, y=9))

    def test_metric_with_two_arguments(self):
        def metric_fn(arg1, arg2): return arg1 - arg2

        metric = Metric("metric", metric_fn, ["x", "y"], None)
        self.assertEqual(-6, metric(x=3, y=9))

    def test_metric_with_transform(self):
        def metric_fn(arg1, arg2): return arg1 - arg2

        def transform(x):
            return x + 1, x - 1

        metric = Metric("metric", metric_fn, ["x"], transform_fn=transform)
        self.assertEqual(2, metric(x=3))
