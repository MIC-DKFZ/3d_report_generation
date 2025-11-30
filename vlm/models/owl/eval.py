import os
import json
from pathlib import Path
from Eval.utils import set_seed
from tqdm import tqdm

from Eval.metrics import BERTScore, RateScore, BleuScoreHF, GREENScore, ROUGEScore, METEORScore

def get_scores_individual(input_data, eval_file):

    results = []
    for sample in tqdm(input_data):
        pred = sample['output_text']
        gt = sample['gt_text']

        rate_score = RateScore([pred], [gt])
        bleu_score = BleuScoreHF([pred], [gt])
        bert_score = BERTScore([pred], [gt])
        # green_score = GREENScore([pred], [gt])
        rouge_score = ROUGEScore([pred], [gt])
        meteor_score = METEORScore([pred], [gt])
        results.append(
            {   
                'img_path': sample['img_path'],
                'input_text': sample['input_text'],
                "prediction": pred,
                "gt": gt,
                'rate_score': rate_score[0],
                "bleu_score_hf": bleu_score['bleu'],
                "bert_score": bert_score['f1'][0],
                # "green_score": green_score,
                "rouge_score": rouge_score['rougeLsum'],
                "meteor_score": meteor_score['meteor']
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
    green_score = GREENScore(pred, gt)
    rouge_score = ROUGEScore(pred, gt)
    meteor_score = METEORScore(pred, gt)

    results = [
        {
            "prediction": pred,
            "gt": gt,
            'rate_score': sum(rate_score) / len(rate_score), # returns the list of scores, we need to take the mean
            "bleu_score_hf": bleu_score['bleu'], # returns the avg,
            "bert_score": sum(bert_score['f1']) / len(bert_score['f1']), # returns the list of scores, we need to take the mean
            "green_score": green_score,
            "rouge_score": rouge_score['rougeLsum'],
            "meteor_score": meteor_score['meteor']
        }
    ]

    if eval_file is not None:
        # save evaluation as JSON
        if not Path(eval_file).parent.is_dir():
            os.makedirs(Path(eval_file).parent)
        with open(eval_file, 'w') as json_file:
            json.dump(results, json_file, indent=4)

def main(input_data, output_dir, seed = 123):
    set_seed(seed)
    get_scores_individual(input_data=input_data, eval_file=output_dir)
    # get_scores_avg(input_data=input_data, eval_file=Path('./owl_results/eval_results_owl.json'))


if __name__ == '__main__':
    input_file_path = Path('./owl/results/inference_results_owl.json')
    output_dir = Path('./owl/results/eval_results_owl.json')
    with open(input_file_path, 'r') as json_file:
        input_data = json.load(json_file)

    main(input_data, output_dir)