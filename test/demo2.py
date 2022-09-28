from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
import numpy as np
ctxs = [mx.gpu(0)] # or, e.g., [mx.gpu(0), mx.gpu(1)]

# MXNet MLMs (use names from mlm.models.SUPPORTED_MLMS)
model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
scorer = MLMScorer(model, vocab, tokenizer, ctxs)

choices = ["apart", "different", "far", "free"]
sentences = [f"Due to the rain, our performance in the game was {choice} from perfect." for choice in choices]
result = np.array(scorer.score_sentences(sentences))
result_index = result.argsort()[::-1]

print("| 文 | スコア |")
print("| ---- | ---- |")

for index in result_index:
    print(f"| {sentences[index]} | {round(result[index], 2)} |")