import os

import nltk
from .base import Command
from lafhterlearn.ngrams import build_ngram_model


class CreateWordDistributionCommand(Command):
    name = 'learn_ngrams'
    help = 'Build n-grams using a plain text corpora'

    def configure_parser(self, parser):
        parser.add_argument('output_file', type=str, help='Location of output dictionary file')

        parser.add_argument('corpora_dir', type=str, default='',
                            help='Location of a directory containing plain text files')

        parser.add_argument('--n', type=int, default=2,
                            help='Order of n-gram model (1 for unigrams, 2 for bigrams, etc.). 2 by default')

        parser.add_argument('--max-words', type=int, default=10**12,
                            help='Maximum number of words to consider in text corpora before stopping. '
                                 'Consider all words by default')

        parser.add_argument('--only-letters', default=False, action='store_true',
                            help='Filters out words containing digits and other non-letter characters')

    def __call__(self, args):
        corpora_dir = args.corpora_dir
        max_words = args.max_words
        n = args.n

        def get_corpus():
            return sentences_stream(corpora_dir, max_words)
        return build_ngram_model(get_corpus, args.output_file, n=n)


def load_sentences(corpora_dir):
    """A generator yielding words contained in text files located within corpora_dir folder"""
    for text_file in os.listdir(corpora_dir):
        path = os.path.join(corpora_dir, text_file)
        with open(path) as f:
            text = f.read().strip()
            for sent in nltk.sent_tokenize(text):
                yield nltk.word_tokenize(sent)


def sentences_stream(corpora_dir, max_words):
    gen = load_sentences(corpora_dir)
    gen = data_limiter(gen, max_words)
    return with_progress_bar(gen)


def data_limiter(sentences, max_words):
    words = 0
    for sent in sentences:
        if words > max_words:
            break

        words += len(sent)
        yield sent


def with_progress_bar(it):
    for i, elem in enumerate(it):
        if i % 100 == 0:
            print(f'\rProcessed sentences: {i + 1}', end='')
        yield elem
    print()
