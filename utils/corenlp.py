from stanza.server import CoreNLPClient

tokenizer_client = CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=30000, memory='16G', threads=16, properties={'tokenize.ptb3Escaping': False, 'ssplit.eolonly': True, 'tokenize.options': "splitHyphenated=true,invertible=true"})
def bulk_tokenize(text, return_offsets=False):
    ann = tokenizer_client.annotate('\n'.join(text))

    if return_offsets:
        return [[token.originalText for token in sentence.token] for sentence in ann.sentence], [[(token.beginChar, token.endChar) for token in sentence.token] for sentence in ann.sentence]
    else:
        return [[token.originalText for token in sentence.token] for sentence in ann.sentence]

if __name__ == "__main__":
    print(bulk_tokenize(['This is a test sentence.', ' This is another.']))
