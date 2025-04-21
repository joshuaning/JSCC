import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer
import pandas as pd
import numpy as np


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

    # a = compute_bleu(
    #     ""abbiamo tentato di riportarla nel limburgo , a maastricht , città che tutti voi conoscete .",
    #     "abbiamo tentato di kittelmann nel kittelmann , a maastricht , città che tutti voi conoscete .")
    
    # print(a)

    gt_fname = 'inference_results\single_lang_enc_it_dec_gt.csv'
    pred_fname = 'inference_results\single_lang_enc_it_dec_pred.csv'
    gt_df = pd.read_csv(gt_fname, header=None)
    pred_df = pd.read_csv(pred_fname, header=None)
    all_wer = []
    all_bleu = []
    # print(len(pred_df))
    # print(gt_df)

    for i in range(len(pred_df)):
        cur_pred = pred_df.iloc[i][0]
        cur_gt = gt_df.iloc[i][0]
        all_bleu.append(compute_bleu(cur_gt, cur_pred))
        all_wer.append(compute_wer(cur_gt, cur_pred))
        # print(cur_pred)
        # print(cur_gt)
        # print("cur_bleu = ", compute_bleu(cur_gt, cur_pred))
        # print("cur_wer = ", compute_wer(cur_gt, cur_pred))
        # break
    
    print("avg bleu score: ", np.mean(np.array(all_bleu)))
    print("avg wer score: ", np.mean(np.array(all_wer)))