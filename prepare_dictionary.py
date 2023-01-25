import argparse
import csv

from nltk.probability import FreqDist
from nltk.corpus import gutenberg


def load_words(dict_file):
    with open(dict_file) as f:
        return [word.strip() for word in f if word.strip()]


def filter_words(words, only_letters=True):
    max_word_len = args.max_len
    words = [w for w in words if len(w.strip()) <= max_word_len]

    if only_letters:
        words = [w for w in words if w.isalpha()]
    return words


def save_distr(words, destination):
    fdist = FreqDist()

    for word in words:
        fdist[word] += 1

    with open(destination, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for word in fdist.elements():
            p = fdist.freq(word)
            writer.writerow([word, p])


def save_dict_file(words, destination):
    unique = set(words)
    with open(destination, 'w') as f:
        file_content = '\n'.join(unique)
        f.write(file_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare dictionary file used to synthesize word images'
    )

    parser.add_argument('output_file', type=str, help='Location of output dictionary file')

    parser.add_argument('--dict-file', type=str, default='',
                        help='Location of a dictionary file (required if --gutenberg is omitted)')

    parser.add_argument('--gutenberg', default=False, action='store_true',
                        help='When set, will use NLTK corpus (gutenberg) to extract words.')
    parser.add_argument('--max-len', type=int, default=14,
                        help='Exclude words from the output file longer than max-len characters')
    parser.add_argument('--only-letters', default=False, action='store_true',
                        help='Filters out words containing digits and other non-letter characters')
    parser.add_argument('--with-freq', default=False, action='store_true',
                        help='Whether to calculate frequencies of each word')

    args = parser.parse_args()

    all_words = gutenberg.words() if args.gutenberg else load_words(args.dict_file)

    save_func = save_distr if args.with_freq else save_dict_file
    save_func(all_words, args.output_file)
