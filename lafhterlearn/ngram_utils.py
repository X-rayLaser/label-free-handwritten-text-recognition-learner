from collections import OrderedDict

import h5py
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams
import numpy as np


def load_sparse_array(indices, values, size):
    a = SparseArray(size)
    for i, v in zip(indices, values):
        a[i] = v
    return a


class SparseArray:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill
        self.data = OrderedDict()

    @property
    def indices(self):
        return list(self.data.keys())

    @property
    def values(self):
        return list(self.data.values())

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if 0 <= idx < len(self):
            return self.data.get(idx, self.fill)

        raise IndexError(f'index out of bounds: {idx}')

    def __setitem__(self, key, value):
        if 0 <= key < len(self):
            self.data[key] = value
        else:
            raise IndexError(f'index out of bounds: {key}')


class SparseMatrix:
    """Resizable ragged matrix with sparse rows"""
    def __init__(self, h5obj, group_name, cols):
        if group_name not in h5obj:
            group = h5obj.create_group(group_name)
        else:
            group = h5obj[group_name]

        self.group = group
        self.cols = cols

        indices_name = "indices"
        values_name = "values"

        self.indices = RaggedMatrix(group, indices_name)
        self.values = RaggedMatrix(group, values_name)

    def add_empty_row(self):
        self.indices.append_row([])
        self.values.append_row([])

    def bulk_update(self, row, indices, values):
        """Replace/put values in specified positions in a given row"""
        sparse_array = self[row]
        for idx, value in zip(indices, values):
            sparse_array[idx] = value

        self.indices.update_row(row, sparse_array.indices)
        self.values.update_row(row, sparse_array.values)

    def __getitem__(self, idx):
        indices_row = self.indices[idx]
        values_row = self.values[idx]
        return load_sparse_array(indices_row, values_row, self.cols)

    def __len__(self):
        return len(self.indices)


class ProbabilityDistribution:
    def __init__(self, sparse_counts_array):
        self.sparse_counts_array = sparse_counts_array
        self.rng = np.random.default_rng()

    def sample(self):
        counts = self.sparse_counts_array.values
        pmf = counts / sum(counts)

        indices = list(range(counts))
        return self.rng.choice(indices, p=pmf)


class BaseH5Matrix:
    def __init__(self, h5obj, ds_name, create_dataset):
        if ds_name not in h5obj:
            dset = create_dataset()
        else:
            dset = h5obj[ds_name]

        self.dset = dset

    def append_row(self, arr):
        current_row = len(self)
        self.dset.resize(current_row + 1, axis=0)
        self.dset[current_row] = arr[:]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if 0 <= idx < len(self):
                return self.dset[idx]
            raise IndexError()

        raise IndexError('index should be an integer, not a tuple')

    def __len__(self):
        return len(self.dset)


class ExpandableMatrix(BaseH5Matrix):
    """Matrix backed by H5 with fixed number of columns, but dynamic number of rows"""
    def __init__(self, h5obj, ds_name, cols):
        def create_dataset():
            return h5obj.create_dataset(ds_name, (0, cols), maxshape=(None, cols), dtype='i')
        super().__init__(h5obj, ds_name, create_dataset)
        self.cols = cols

    def append_row(self, arr):
        if len(arr) != self.cols:
            raise ValueError(f'expects rows of length {self.cols}, got {len(arr)}')
        super().append_row(arr)


class RaggedMatrix(BaseH5Matrix):
    """Resizable matrix with varying number of columns"""
    def __init__(self, h5obj, ds_name):
        def create_dataset():
            dt = h5py.vlen_dtype(np.dtype('int32'))
            return h5obj.create_dataset(ds_name, (0,), maxshape=(None,), dtype=dt)
        super().__init__(h5obj, ds_name, create_dataset)

    def update_row(self, row, values):
        self.dset[row] = values[:]


def build_vocab(words, unk_cutoff=10):
    from collections import Counter
    counter = Counter(words)
    vocab = set(word for word, count in counter.items() if count >= unk_cutoff)
    vocab.add('<UNK>')
    vocab.add('<s>')
    vocab.add(r'<\s>')
    return vocab


def get_word_stream(get_corpus, n):
    for sent in get_corpus():
        yield from pad_both_ends(sent, n=n)


def get_ngram_stream(get_corpus, n):
    return ngrams(get_word_stream(get_corpus, n), n=n)
