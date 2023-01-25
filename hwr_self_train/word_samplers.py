import random
import csv
import numpy as np


class UniformSampler:
    def __init__(self, words, max_word_len=14):
        self.words = words

    def __call__(self):
        return random.choice(self.words)

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            words = [word.strip() for word in f if word.strip()]
            return cls(words)


class FrequencyBasedSampler:
    def __init__(self, words, frequencies):
        self.words = words
        self.frequencies = frequencies

    def __call__(self):
        return np.random.choice(self.words, p=self.frequencies)

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
