import torch


class ValuePreprocessor:
    def fit(self, dataset):
        pass

    def process(self, value):
        pass

    def __call__(self, value):
        return self.process(value)

    def wrap_preprocessor(self, preprocessor):
        """Wraps a given preprocessor with self.

        Order of preprocessing: pass a value through a new preprocessor,
        then preprocess the result with self

        :param preprocessor: preprocessor to wrap
        :return: A callable
        """
        return lambda value: self.process(preprocessor(value))


class CharacterTokenizer(ValuePreprocessor):
    start = '<s>'
    end = r'<\s>'
    out_of_charset = '<?>'

    english = "english"

    def __init__(self, charset=None):
        charset = charset or self.english

        if charset == self.english:
            letters = "abcdefghijklmnopqrstuvwxyz"
            digits = "0123456789"
            punctuation = ".,?!:;-()'\""
            letters.upper()
            charset = letters + letters.upper() + digits + punctuation

        self.char2index = {}
        self.index2char = {}
        self.charset = charset
        self._build_char_table(charset)

    def process(self, text):
        start_token = self._encode(self.start)
        end_token = self._encode(self.end)
        tokens = [self._encode(ch) for ch in text]
        return [start_token] + tokens + [end_token]

    @property
    def charset_size(self):
        return len(self.char2index)

    def _build_char_table(self, charset):
        self._add_char(self.start)
        self._add_char(self.end)
        self._add_char(self.out_of_charset)

        for ch in charset:
            self._add_char(ch)

    def _add_char(self, ch):
        if ch not in self.char2index:
            num_chars = self.charset_size
            self.char2index[ch] = num_chars
            self.index2char[num_chars] = ch

    def _encode(self, ch):
        default_char = self.char2index[self.out_of_charset]
        return self.char2index.get(ch, default_char)

    def _decode(self, token):
        return self.index2char.get(token, self.out_of_charset)

    def decode_to_string(self, tokens, clean_output=False):
        s = ''.join([self._decode(token) for token in tokens[1:-1]])

        first_char = self._decode(tokens[0])
        last_char = self._decode(tokens[-1])

        if first_char != self.start:
            s = first_char + s

        if last_char != self.end:
            s += last_char

        if clean_output:
            s = s.replace(self.start, '')
            try:
                sentinel_idx = s.index(self.end)
                s = s[:sentinel_idx]
            except ValueError:
                pass

            s = s.replace(self.out_of_charset, '')

        return s


class DecodeCharacterString:
    def __init__(self, session):
        self.tokenizer = session.preprocessors["tokenize"]

    def __call__(self, y_hat, ground_true):
        if type(ground_true) is torch.Tensor:
            ground_true = ground_true.tolist()

        y_hat = y_hat.argmax(dim=2).tolist()
        tokenizer = self.tokenizer
        predicted_texts = []
        actual_texts = []

        for predicted_tokens, true_tokens in zip(y_hat, ground_true):
            predicted_texts.append(tokenizer.decode_to_string(predicted_tokens, clean_output=True))
            actual_texts.append(tokenizer.decode_to_string(true_tokens, clean_output=True))

        return predicted_texts, actual_texts


def decode_output_batch(tensor, tokenizer):
    """Convert tensor of predictions into a list of strings.

    :param tensor: tensor of shape (batch_size, max_steps, num_classes) containing
    raw (unnormalized) scores representing how likely is a given character at a given step

    :param tokenizer: instance of CharacterTokenizer class
    :return: textual transcripts extracted using predictions tensor
    """

    y_hat = tensor.argmax(dim=2).tolist()
    return [tokenizer.decode_to_string(token_list, clean_output=True)
            for token_list in y_hat]
