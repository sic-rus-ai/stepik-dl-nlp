import spacy_udpipe
from .base import NUMERIC_RE, NUMERIC_TOKEN


def setup_spacy(language='ru'):
    spacy_udpipe.download(language)


UNKNOWN_PLACEHOLDER_TEMPLATE = '__UNK_{}__'
UNKNOWN_PLACEHOLDER_TOKENS = [
    UNKNOWN_PLACEHOLDER_TEMPLATE.format(pos_tag)
    for pos_tag in 'ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X'.split(' ')
]


class SpacyCorpusLemmatizer:
    def __init__(self, language='ru', add_postags=True, vocabulary=None):
        self.pipeline = spacy_udpipe.load(language)
        self.add_postags = add_postags
        self.vocabulary = vocabulary

    def __call__(self, texts):
        parsed = self.pipeline.pipe(texts)
        return [self.make_tokens(par) for par in parsed]

    def make_tokens(self, parse_result):
        filtered_tokens = []
        for token in parse_result:
            if NUMERIC_RE.match(token.text):
                token_text = NUMERIC_TOKEN
            else:
                token_text = token.lemma_.lower()
                if self.add_postags:
                    token_text += '_' + token.pos_

                if self.vocabulary is not None:
                    if token_text not in self.vocabulary:
                        token_text = UNKNOWN_PLACEHOLDER_TEMPLATE.format(token.pos_)

            filtered_tokens.append((token_text, (token.idx, token.idx + len(token.text))))

        return filtered_tokens
