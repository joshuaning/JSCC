import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer


# Compute the BLEU score for a single sentence.s
def compute_bleu(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.strip().split()
    hypothesis_tokens = hypothesis.strip().split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    return score

#Compute the Word Error Rate (WER) for a single sentence.
def compute_wer(reference: str, hypothesis: str) -> float:
    return wer(reference, hypothesis)



if __name__ == "__main__":

    a = 1
## tested with 
'''
python eval.py --reference "The quick brown fox jumps over the lazy dog" \
               --hypothesis "The quick brown fox jump over a lazy dog"
'''