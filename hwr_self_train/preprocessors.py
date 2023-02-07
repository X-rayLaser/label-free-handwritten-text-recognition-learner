class CharacterTokenizer:
    start = '<s>'
    end = r'<\s>'
    out_of_charset = '<?>'

    english = "english"

    def __init__(self, charset):
        self.char2index = {}
        self.index2char = {}
        self.charset = charset
        self._build_char_table(charset)

    def __call__(self, value):
        return self.process(value)

    def process(self, text):
        start_token = self.encode(self.start)
        end_token = self.encode(self.end)
        tokens = [self.encode(ch) for ch in text]
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

    def encode(self, ch):
        default_char = self.char2index[self.out_of_charset]
        return self.char2index.get(ch, default_char)

    def decode(self, token):
        return self.index2char.get(token, self.out_of_charset)

    def decode_to_string(self, tokens, clean_output=False):
        s = ''.join([self.decode(token) for token in tokens[1:-1]])

        first_char = self.decode(tokens[0])
        last_char = self.decode(tokens[-1])

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
