import itertools
from unittest import TestCase
import h5py
import os
import numpy as np

from lafhterlearn.metrics import Metric
from lafhterlearn.ngram_utils import ExpandableMatrix, RaggedMatrix, SparseMatrix, backoff
from lafhterlearn.ngrams import chunks, sorted_chunks, merge_chunks, bigsort, ChainSequence


class BackoffTests(TestCase):
    def test(self):
        self.assertEqual([], list(backoff(lam=0.4)))

        self.assertEqual([], list(backoff([], lam=0.4)))

        self.assertEqual([1], list(backoff([1], lam=0.4)))

        self.assertEqual([0., 0.4, 0., 0.6], list(backoff([0, 0.4, 0, 0.6], lam=0.4)))
        self.assertEqual([0., 0.4, 0., 0.6], list(backoff([0, 0.4, 0, 0.6], lam=0.7)))

        self.assertEqual([1], list(backoff([1], [1], lam=0.4)))
        self.assertEqual([1], list(backoff([1], [0], lam=0.4)))
        self.assertEqual([0.4], list(backoff([0], [1], lam=0.4)))
        self.assertEqual([0], list(backoff([0], [0], lam=0.4)))

        self.assertEqual([0.4, 1.], list(backoff([0, 1], [1, 0], lam=0.4)))
        self.assertEqual([0., 1.], list(backoff([0, 1], [0, 1], lam=0.4)))

        probs = backoff([0, 1], [0.2, 0.8], lam=0.4)
        self.assertTrue(np.allclose(np.array([0.08, 1.]), probs))

        probs = backoff([0, 0.5, 0.3, 0, 0.2], [0, 0, 0.4, 0.5, 0.1], lam=0.4)
        self.assertTrue(np.allclose([0.0, 0.5, 0.3, 0.2, 0.2], probs))

        probs = backoff([1, 0], [1, 0], [1, 0], lam=0.4)
        self.assertEqual([1, 0], list(probs))

        probs = backoff([1, 0], [0, 1], [1, 0], lam=0.4)
        self.assertEqual([1, 0.4], list(probs))
        probs = backoff([1, 0], [0, 1], [0, 1], lam=0.4)
        self.assertEqual([1, 0.4], list(probs))

        probs = backoff([1, 0], [1, 0], [0, 1], lam=0.4)
        self.assertEqual([1, 0.4**2], list(probs))
        probs = backoff([1, 0], [1, 0], [0.2, 0.8], lam=0.4)
        self.assertEqual([1, 0.8 * 0.4**2], list(probs))


class ChainSequenceTests(TestCase):
    def test_len(self):
        seq = ChainSequence()
        self.assertEqual(0, len(seq))

        seq = ChainSequence([])
        self.assertEqual(0, len(seq))

        seq = ChainSequence([], [], [])
        self.assertEqual(0, len(seq))

        seq = ChainSequence([1, 2, 3])
        self.assertEqual(3, len(seq))

        seq = ChainSequence([1, 2], [3, 4, 5])
        self.assertEqual(5, len(seq))

        seq = ChainSequence([1, 2, 3], [0], [], [0])
        self.assertEqual(5, len(seq))

    def test_indexing(self):
        seq = ChainSequence()
        self.assertRaises(IndexError, lambda: seq[0])

        seq = ChainSequence([])
        self.assertRaises(IndexError, lambda: seq[0])

        seq = ChainSequence([], [], [])
        self.assertRaises(IndexError, lambda: seq[0])

        seq = ChainSequence([30])
        self.assertEqual(30, seq[0])

        seq = ChainSequence([5, 3], [10, 20, 30])
        self.assertEqual(5, seq[0])
        self.assertEqual(3, seq[1])
        self.assertEqual(10, seq[2])
        self.assertEqual(20, seq[3])
        self.assertEqual(30, seq[4])

        seq = ChainSequence([2], [0], [12], [99], [33])
        self.assertEqual(2, seq[0])
        self.assertEqual(0, seq[1])
        self.assertEqual(12, seq[2])
        self.assertEqual(99, seq[3])
        self.assertEqual(33, seq[4])

        seq = ChainSequence([1, 2], [5], [12, 17, 20, 34], [99])
        self.assertEqual(1, seq[0])
        self.assertEqual(2, seq[1])
        self.assertEqual(5, seq[2])
        self.assertEqual(12, seq[3])
        self.assertEqual(17, seq[4])
        self.assertEqual(20, seq[5])
        self.assertEqual(34, seq[6])
        self.assertEqual(99, seq[7])


class ChunkingTests(TestCase):
    def test(self):
        self.assertRaises(ValueError, lambda: list(chunks([1, 2, 3], 0)))

        self.assertEqual([], list(chunks([], 5)))

        self.assertEqual([[1]], list(chunks([1], 1)))
        self.assertEqual([[1]], list(chunks([1], 2)))

        self.assertEqual([[1], [2]], list(chunks([1, 2], 1)))
        self.assertEqual([[1, 2]], list(chunks([1, 2], 2)))
        self.assertEqual([[1, 2]], list(chunks([1, 2], 3)))

        self.assertEqual([[1], [2], [3]], list(chunks([1, 2, 3], 1)))
        self.assertEqual([[1, 2], [3]], list(chunks([1, 2, 3], 2)))
        self.assertEqual([[1, 2, 3]], list(chunks([1, 2, 3], 3)))
        self.assertEqual([[1, 2, 3]], list(chunks([1, 2, 3], 4)))

        self.assertEqual([[1], [2], [3], [4]], list(chunks([1, 2, 3, 4], 1)))
        self.assertEqual([[1, 2], [3, 4]], list(chunks([1, 2, 3, 4], 2)))
        self.assertEqual([[1, 2, 3], [4]], list(chunks([1, 2, 3, 4], 3)))
        self.assertEqual([[1, 2, 3, 4]], list(chunks([1, 2, 3, 4], 4)))


class SortedChunksTests(TestCase):
    def test(self):
        self.assertEqual([[1, 2, 3], [2, 3, 4]], list(sorted_chunks([3, 2, 1, 2, 4, 3], 3)))
        self.assertEqual([[1, 2, 3], [2, 4]], list(sorted_chunks([3, 2, 1, 2, 4], 3)))
        self.assertEqual([[1, 2, 2, 3], [4]], list(sorted_chunks([3, 2, 1, 2, 4], 4)))
        self.assertEqual([[3], [2], [1], [2]], list(sorted_chunks([3, 2, 1, 2], 1)))


class MergeChunksTests(TestCase):
    def test(self):
        self.assertEqual([1, 3, 4, 4, 5, 5], list(merge_chunks([1, 3, 4, 4, 5, 5])))

        self.assertEqual([], list(merge_chunks([])))
        self.assertEqual([], list(merge_chunks([], [])))
        self.assertEqual([], list(merge_chunks([], [], [], [], [], [])))

        self.assertEqual([1, 2], list(merge_chunks([1, 2], [])))
        self.assertEqual([-2, 0, 1, 2], list(merge_chunks([], [], [1, 2], [], [-2, 0])))

        self.assertEqual([1, 2, 3, 4], list(merge_chunks([1, 2], [3, 4])))
        self.assertEqual([1, 2, 3, 4], list(merge_chunks([3, 4], [1, 2])))

        self.assertEqual([0, 1, 1, 2, 3, 4, 4, 6], list(merge_chunks([1, 3, 4], [0, 1, 2, 4, 6])))

        merged = merge_chunks([1, 2, 4, 7, 8], [3, 5, 6, 9])
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8, 9], list(merged))

        merged = merge_chunks([5, 9], [3, 3, 4], [-2, 0, 4, 5], [1])
        self.assertEqual([-2, 0, 1, 3, 3, 4, 4, 5, 5, 9], list(merged))

        merged = merge_chunks([5], [4], [3], [2], [1])
        self.assertEqual([1, 2, 3, 4, 5], list(merged))

        merged = merge_chunks([1], [2], [3], [4], [5])
        self.assertEqual([1, 2, 3, 4, 5], list(merged))

        merged = merge_chunks([1, 2], [3, 4, 5])
        self.assertEqual([1, 2, 3, 4, 5], list(merged))

        p = next(itertools.permutations(range(100)))

        merged = merge_chunks(*list(sorted_chunks(p, size=23)))
        self.assertEqual(list(range(100)), list(merged))


class BigsortTests(TestCase):
    def setUp(self) -> None:
        if os.path.isfile("chunks.h5"):
            os.remove("chunks.h5")

    def tearDown(self) -> None:
        if os.path.isfile("chunks.h5"):
            os.remove("chunks.h5")

    def test_sort_integers(self):
        gen = bigsort([2, 4, 3, 5, 6, 7, 1, 0, 8, 9], chunk_size=3)
        self.assertEqual(list(range(10)), list(gen))

    def test_sort_numeric_tuples(self):
        gen = bigsort([(3, 5), (1, 3), (2, 4), (1, 2), (3, 1)], chunk_size=3)
        self.assertEqual([(1, 2), (1, 3), (2, 4), (3, 1), (3, 5)], list(gen))


class ExpandableMatrixTests(TestCase):
    file_name = "testfile.h5"

    def setUp(self) -> None:
        f = h5py.File(self.file_name, mode='w')
        f.close()

    def tearDown(self) -> None:
        os.remove(self.file_name)

    def test_initial(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = ExpandableMatrix(f, "ds", 2)
            self.assertEqual(0, len(matrix))

    def test_read_from_empty(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = ExpandableMatrix(f, "ds", 2)
            self.assertRaises(IndexError, lambda: matrix[0])

    def test_add_single_row(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = ExpandableMatrix(f, "ds", 3)

            self.assertRaises(ValueError, lambda: matrix.append_row([1]))
            self.assertRaises(ValueError, lambda: matrix.append_row([1, 10]))
            self.assertRaises(ValueError, lambda: matrix.append_row([1, 10, 100, 10]))

            matrix.append_row([1, 2, 4])
            self.assertEqual(1, len(matrix))
            self.assertEqual([1, 2, 4], list(matrix[0]))

    def test_two_rows(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = ExpandableMatrix(f, "ds", 2)
            matrix.append_row([5, 10])
            matrix.append_row([20, 30])

            self.assertEqual(2, len(matrix))
            self.assertEqual([5, 10], list(matrix[0]))
            self.assertEqual([20, 30], list(matrix[1]))


class H5MatrixTests(TestCase):
    file_name = "testfile.h5"

    def setUp(self) -> None:
        f = h5py.File(self.file_name, mode='w')
        f.close()

    def tearDown(self) -> None:
        os.remove(self.file_name)

    def test_initial(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = RaggedMatrix(f, "ds")
            self.assertEqual(0, len(matrix))

    def test_read_from_empty(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = RaggedMatrix(f, "ds")
            self.assertRaises(IndexError, lambda: matrix[0])
            self.assertRaises(IndexError, lambda: matrix[20])
            self.assertRaises(IndexError, lambda: matrix[0, 0])

    def test_add_single_row(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = RaggedMatrix(f, "ds")
            matrix.append_row([1, 2, 4])
            self.assertEqual(1, len(matrix))
            self.assertEqual([1, 2, 4], list(matrix[0]))

    def test_add_empty_row(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = RaggedMatrix(f, "ds")
            matrix.append_row([])
            self.assertEqual(1, len(matrix))
            self.assertEqual([], list(matrix[0]))

    def test_add_few_rows_of_different_length(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = RaggedMatrix(f, "ds")
            matrix.append_row([1, 2, 4])
            matrix.append_row([5, 6])
            matrix.append_row([9, 2, 0, 5])

            self.assertEqual(3, len(matrix))
            self.assertEqual([1, 2, 4], list(matrix[0]))
            self.assertEqual([5, 6], list(matrix[1]))
            self.assertEqual([9, 2, 0, 5], list(matrix[2]))

    def test_update_row(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = RaggedMatrix(f, "ds")
            matrix.append_row([1, 2, 4])
            matrix.append_row([5, 6])

            matrix.update_row(1, [3, 2, 5, 10])
            self.assertEqual(2, len(matrix))
            self.assertEqual([1, 2, 4], list(matrix[0]))
            self.assertEqual([3, 2, 5, 10], list(matrix[1]))

    def test_persistence(self):
        with h5py.File(self.file_name, mode="r+") as f:
            matrix = RaggedMatrix(f, "ds")
            matrix.append_row([1, 2, 4])

            del matrix
            new_matrix = RaggedMatrix(f, "ds")
            self.assertEqual(1, len(new_matrix))
            self.assertEqual([1, 2, 4], list(new_matrix[0]))


class SparseMatrixTests(TestCase):
    file_name = "test_file_sparse.h5"

    def setUp(self) -> None:
        f = h5py.File(self.file_name, mode='w')
        f.close()

    def tearDown(self) -> None:
        os.remove(self.file_name)

    def test_initial(self):
        with h5py.File(self.file_name, mode='w') as f:
            matrix = SparseMatrix(f, "group", 4)
            self.assertEqual(len(matrix), 0)

    def test_append_row(self):
        with h5py.File(self.file_name, mode='w') as f:
            matrix = SparseMatrix(f, "group", 5)
            matrix.add_empty_row()
            self.assertEqual(len(matrix), 1)
            self.assertEqual([0, 0, 0, 0, 0], list(matrix[0]))

            matrix.bulk_update(0, [0, 2, 3], [9, 8, 7])
            self.assertEqual(len(matrix), 1)
            self.assertEqual([9, 0, 8, 7, 0], list(matrix[0]))

    def test_append_2_rows(self):
        with h5py.File(self.file_name, mode='w') as f:
            matrix = SparseMatrix(f, "group", 4)
            matrix.add_empty_row()
            matrix.add_empty_row()
            matrix.bulk_update(0, [1, 2], [5, 10])
            matrix.bulk_update(1, [0, 2], [5, 10])

            self.assertEqual([0, 5, 10, 0], list(matrix[0]))
            self.assertEqual([5, 0, 10, 0], list(matrix[1]))

    def test_row_updates(self):
        with h5py.File(self.file_name, mode='w') as f:
            matrix = SparseMatrix(f, "group", 4)
            matrix.add_empty_row()
            matrix.add_empty_row()
            matrix.add_empty_row()
            matrix.bulk_update(1, [1, 2], [5, 10])
            matrix.bulk_update(1, [0, 2, 3], [100, 20, 50])
            self.assertEqual([0, 0, 0, 0], list(matrix[0]))
            self.assertEqual([100, 5, 20, 50], list(matrix[1]))
            self.assertEqual([0, 0, 0, 0], list(matrix[2]))

    def test_cannot_use_wrong_indices(self):
        with h5py.File(self.file_name, mode='w') as f:
            matrix = SparseMatrix(f, "group", 4)
            matrix.add_empty_row()
            self.assertRaises(IndexError, lambda: matrix.bulk_update(3, [3], [3]))
            self.assertRaises(IndexError, lambda: matrix.bulk_update(0, [1, 2, 6], [9, 8, 7]))


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
