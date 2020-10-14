import re
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])

WHITESPACE = re.compile('\s+')

def bulk_tokenize(text, return_offsets=False):
    ann = list(nlp.pipe(text))

    if return_offsets:
        return [[w.text for w in s if not WHITESPACE.match(w.text)] for s in ann], [[(w.idx, w.idx+len(w.text)) for w in s if not WHITESPACE.match(w.text)] for s in ann]
    else:
        return [[w.text for w in s if not WHITESPACE.match(w.text)] for s in ann]

    return ann

if __name__ == "__main__":
    print(bulk_tokenize(['This is a test sentence.', 'This is another. Actually, two sentences.'], return_offsets=True))
