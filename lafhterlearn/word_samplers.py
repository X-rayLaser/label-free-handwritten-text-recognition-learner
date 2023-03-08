import random
import csv
import string

import numpy as np
import h5py

from .ngrams import NgramModel, ModelProxy


class UniformSampler:
    def __init__(self, path):
        with open(path) as f:
            self.words = [word.strip() for word in f if word.strip()]

    def __call__(self):
        return random.choice(self.words)


class FrequencyBasedSampler:
    def __init__(self, path, sampling_batch_size=1024):
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            words = []
            frequencies = []
            for word, freq in reader:
                words.append(word)
                frequencies.append(freq)

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


class NgramBasedSampler:
    def __init__(self, path, num_words=10):
        self.path = path
        self.proxy = ModelProxy(path)
        self.num_words = num_words

    def __call__(self):
        words = self.proxy.generate(num_words=self.num_words)
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
