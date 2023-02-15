import csv
import os

import nltk
from nltk.probability import FreqDist

from .base import Command


class CreateWordDistributionCommand(Command):
    name = 'word_distr'
    help = 'Prepare dictionary file or word distribution used to synthesize word images'

    def configure_parser(self, parser):
        configure_parser(parser)

    def __call__(self, args):
        run(args)


def configure_parser(parser):
    parser.add_argument('output_file', type=str, help='Location of output dictionary file')

    parser.add_argument('--dict-file', type=str, default='',
                        help='Location of a dictionary file (required if --corpora-dir is omitted)')

    parser.add_argument('--corpora-dir', type=str, default='',
                        help='Generate words from a folder of text files')

    parser.add_argument('--max-len', type=int, default=14,
                        help='Exclude words from the output file longer than max-len characters')
    parser.add_argument('--only-letters', default=False, action='store_true',
                        help='Filters out words containing digits and other non-letter characters')
    parser.add_argument('--with-freq', default=False, action='store_true',
                        help='Whether to calculate frequencies of each word')


def run(args):
    words_gen = load_corpora(args.corpora_dir) if args.corpora_dir else load_dict_file(args.dict_file)

    words_gen = filter_words(words_gen, args.only_letters, max_word_len=args.max_len)

    save_func = save_distr if args.with_freq else save_dict_file
    save_func(words_gen, args.output_file)


def load_dict_file(dict_file):
    """A generator yielding words from a dictionary text file"""
    with open(dict_file) as f:
        for word in f:
            if word.strip():
                yield word


def load_corpus(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield from nltk.word_tokenize(line)


def load_corpora(corpora_dir):
    """A generator yielding words contained in text files located within corpora_dir folder"""
    for text_file in os.listdir(corpora_dir):
        path = os.path.join(corpora_dir, text_file)
        yield from load_corpus(path)


def filter_words(words, allow_only_letters=True, max_word_len=14):
    for w in words:
        too_long = len(w.strip()) > max_word_len
        has_non_letters = not w.isalpha()
        if too_long or (allow_only_letters and has_non_letters):
            continue
        yield w


def save_distr(words, destination):
    fdist = FreqDist()

    for i, word in enumerate(words):
        fdist[word] += 1

        if i % 100 == 0:
            print(f'\rCounted words: {i + 1}', end='')

    print('\nFrequency dictionary is complete')

    with open(destination, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for word in fdist.keys():
            p = fdist.freq(word)
            writer.writerow([word, p])


def save_dict_file(words, destination):
    vocab = set()
    for i, w in enumerate(words):
        vocab.add(w)
        if i % 100 == 0:
            print(f'\rBuilding vocab. Words processed: {i + 1}', end='')

    print('\nVocab is built')
    with open(destination, 'w') as f:
        file_content = '\n'.join(vocab)
        f.write(file_content)
