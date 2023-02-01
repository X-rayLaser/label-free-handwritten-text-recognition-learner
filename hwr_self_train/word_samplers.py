import random
import csv
import numpy as np


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
        # todo: potential race condition when using data loaders with num_workers > 0
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
