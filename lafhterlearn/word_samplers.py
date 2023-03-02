import random
import csv
import string

import numpy as np
import h5py

from .ngrams import NgramModel


class UniformSampler:
    def __init__(self, words):
        self.words = words

    def __call__(self):
        return random.choice(self.words)

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            words = [word.strip() for word in f if word.strip()]
            return cls(words)


class FrequencyBasedSampler:
    def __init__(self, words, frequencies, sampling_batch_size=1024):
        self.words = words
        self.frequencies = frequencies
        self.rng = np.random.default_rng()

        self.batch_size = sampling_batch_size
        self.buffer = []

    def __call__(self):
        # potential race condition when using data loaders with num_workers > 0
        if not self.buffer:
            self.buffer = list(self.rng.choice(self.words, size=self.batch_size,
                                               p=self.frequencies))

        return self.buffer.pop()

    @classmethod
    def from_file(cls, path):
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            words = []
            freqs = []
            for word, freq in reader:
                words.append(word)
                freqs.append(freq)

            return cls(words, freqs)


class NgramBasedSampler:
    def __init__(self, ngram_model, num_words=10):
        self.model = ngram_model
        self.num_words = num_words

    def __call__(self):
        # todo: make it faster
        words = self.model.generate(num_words=self.num_words)
        prev = words[0]

        seq = [prev]
        for word in words[1:]:
            if ((prev == "'" and word in ["s", "m", "t", "ve"]) or
                    (prev in string.punctuation and word in string.punctuation) or
                    (prev.isalnum() and word in string.punctuation)):
                seq.append(word)
            else:
                seq.append(' ')
                seq.append(word)
            prev = word
        return ''.join(seq)

    @classmethod
    def from_file(cls, path):
        f = h5py.File(path)
        model = NgramModel.from_h5file(f)
        return cls(model)
