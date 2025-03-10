import os
import json
from pathlib import Path
from utils import set_seed
from tqdm import tqdm

from metrics import BERTScore, RateScore, BleuScoreHF

def get_scores_individual(input_data, eval_file):

    pred_report = input_data[0]
    gt_report = input_data[1]

    results = []
    for pred, gt in tqdm(zip(pred_report, gt_report)):
        rate_score = RateScore([pred], [gt])
        bleu_score = BleuScoreHF([pred], [gt])
        bert_score = BERTScore([pred], [gt])
        results.append(
            {
                "prediction": pred,
                "gt": gt,
                'rate_score': rate_score[0],
                "bleu_score_hf": bleu_score['bleu'],
                "bert_score": bert_score['f1'][0]
            }
        )
    
    if eval_file is not None:
        # save evaluation as JSON
        if not Path(eval_file).parent.is_dir():
            os.makedirs(Path(eval_file).parent)
        with open(eval_file, 'w') as json_file:
            json.dump(results, json_file, indent=4)

def get_scores_avg(input_data, eval_file):

    pred = input_data[0]
    gt = input_data[1]

    rate_score = RateScore(pred, gt)
    bleu_score = BleuScoreHF(pred, gt)
    bert_score = BERTScore(pred, gt)

    results = [
        {
            "prediction": pred,
            "gt": gt,
            'rate_score': sum(rate_score) / len(rate_score), # returns the list of scores, we need to take the mean
            "bleu_score_hf": bleu_score['bleu'], # returns the avg,
            "bert_score": sum(bert_score['f1']) / len(bert_score['f1']), # returns the list of scores, we need to take the mean
        }
    ]

    if eval_file is not None:
        # save evaluation as JSON
        if not Path(eval_file).parent.is_dir():
            os.makedirs(Path(eval_file).parent)
        with open(eval_file, 'w') as json_file:
            json.dump(results, json_file, indent=4)

def main(input_data, seed = 123):
    set_seed(seed)
    get_scores_individual(input_data=input_data, eval_file=Path('./Results/result_single.json'))
    get_scores_avg(input_data=input_data, eval_file=Path('./Results/result_avg.json'))
    

if __name__ == '__main__':
    pred_report = ['There are no intracranial hemorrhages.',
                'The musculature and soft tissues are intact.', 'hello this is a test']
    gt_report = ['There is no finding to suggest intracranial hemorrhage.',
                'The muscle compartments are intact.', 'hello this is a test']
    input_data = (pred_report, gt_report)

    main(input_data)