def word_to_text_id(word):
    return (word.text, int(word.id))

def filter_sent(sent, condition, postprocessor=word_to_text_id):
    t = [w for w in sent.words if condition(w)]
    if len(t) > 0:
        return postprocessor(t[0])
    else:
        return None

def is_wh(word):
    return word.xpos.startswith('W')

def to_tuple(tpl):
    # return ('wh', tpl.get('wh', None), 'pred', tpl.get('pred', None), 'arg', tpl.get('arg', None))#, 'ans', tpl.get('ans', None))
    res = tuple()
    for key in ['wh', 'pred', 'arg']:
        if key in tpl and tpl[key] is not None:
            res += tpl[key]
    return res

def parsed_to_tuple(question, answer=None):
    root = filter_sent(question, lambda w: w.deprel == 'root', postprocessor=lambda x: x)
    # ansroot = filter_sent(answer, lambda w: w.deprel == 'root')
    # tpl = {'ans': ansroot}
    tpl = dict()

    #print(qa['question'])
    #print()
    #question.print_dependencies()
    #print()
    if is_wh(root):
        # root is wh-word
        tpl['wh'] = word_to_text_id(root)
        tpl['pred'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'cop')
        tpl['arg'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'nsubj')
    elif filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'expl') is not None:
        # is there?
        tpl['wh'] = word_to_text_id(root) + filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'expl')
        tpl['arg'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'nsubj')
    elif filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'cop') is not None and root.upos in ['ADJ', 'ADV']:
        wh_word = filter_sent(question, lambda w: w.head == int(root.id) and is_wh(w))
        if wh_word is not None:
            # how old
            tpl['wh'] = wh_word + word_to_text_id(root)
            tpl['pred'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'cop')
            tpl['arg'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel in ['nsubj', 'nsubj:pass'] and not is_wh(w))
        else:
            # is ... good?
            tpl['wh'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'cop')
            tpl['pred'] = word_to_text_id(root)
    else:
        obj_found = False
        tpl['wh'] = filter_sent(question, lambda w: w.head == int(root.id) and is_wh(w))
        if tpl['wh'] is None:
            wh = filter_sent(question, lambda w: is_wh(w), postprocessor=lambda x: x)
            if wh is not None:
                # what instrument
                wh_noun = filter_sent(question, lambda w: wh.head == int(w.id))
                tpl['wh'] = word_to_text_id(wh) + wh_noun
                obj_found = True
            else:
                # yes/no question
                tpl['wh'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'aux')
        tpl['pred'] = word_to_text_id(root)
        if not obj_found:
            tpl['arg'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel == 'obj' and not is_wh(w))
            if tpl['arg'] is None:
                # find obliques or nmods
                tpl['arg'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel in ['obl', 'nmod'] and not is_wh(w))
            if tpl['arg'] is None and root.upos == 'NOUN':
                # find adjective
                # tpl['wh'] = tpl['wh'] if tpl.get('wh', None) is not None else 'be'
                tpl['arg'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel in ['amod'] and not is_wh(w))
            if tpl['arg'] is None and tpl['wh'] is None:
                # find subject
                tpl['arg'] = filter_sent(question, lambda w: w.head == int(root.id) and w.deprel in ['nsubj', 'nsubj:pass'] and not is_wh(w))
    return to_tuple(tpl)
