import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer


# Compute the BLEU score for a single sentence.
def compute_bleu(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.strip().split()
    hypothesis_tokens = hypothesis.strip().split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    return score

#Compute the Word Error Rate (WER) for a single sentence.
def compute_wer(reference: str, hypothesis: str) -> float:
    return wer(reference, hypothesis)

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLEU and WER between reference and hypothesis sentences.")
    parser.add_argument("--reference", type=str, required=True, help="Reference sentence (ground truth).")
    parser.add_argument("--hypothesis", type=str, required=True, help="Hypothesis sentence (model output).")
    args = parser.parse_args()

    bleu = compute_bleu(args.reference, args.hypothesis)
    word_error_rate = compute_wer(args.reference, args.hypothesis)

    print(f"BLEU Score: {bleu:.4f}")
    print(f"Word Error Rate (WER): {word_error_rate:.4f}")

if __name__ == "__main__":
    main()


## tested with 
'''
python eval.py --reference "The quick brown fox jumps over the lazy dog" \
               --hypothesis "The quick brown fox jump over a lazy dog"
'''