from rouge.rouge_scorer import RougeScorer
from rouge.scoring import BootstrapAggregator

def rouge_score(references, preds):
    rouge_scorer = RougeScorer(rouge_types=['rougeL'])
    aggregator = BootstrapAggregator()

    for r, p in zip(references, preds):
        aggregator.add_scores(rouge_scorer.score(r, p))

    res = aggregator.aggregate()

    print(res['rougeL'].low.fmeasure, res['rougeL'].high.fmeasure)

    return res['rougeL'].mid.fmeasure

