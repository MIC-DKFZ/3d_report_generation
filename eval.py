import argparse
import os
import json
from pathlib import Path

import yaml
from Eval.utils import set_seed
from tqdm import tqdm

from Eval.metrics import BERTScore, RateScore, BleuScoreHF, GREENScore, ROUGEScore, METEORScore

def get_scores_individual(inference_results_file, eval_dir):
    # Load inference results from JSON file
    input_data = []
    with open(inference_results_file, 'r') as json_file: # TODO: maybe support other formats later
        input_data = json.load(json_file)

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

    eval_file = os.path.join(eval_dir, "eval_results.json")
    if eval_file is not None:
        # save evaluation as JSON
        if not Path(eval_file).parent.is_dir():
            os.makedirs(Path(eval_file).parent)
        with open(eval_file, 'w') as json_file:
            json.dump(results, json_file, indent=4)

def get_scores_avg(inference_results_file, eval_dir):
    # Load inference results from JSON file
    input_data = []
    with open(inference_results_file, 'r') as json_file: # TODO: maybe support other formats later
        input_data = json.load(json_file)

    pred = [item['output_text'] for item in input_data if 'output_text' in item]
    gt = [item['gt_text'] for item in input_data if 'gt_text' in item]

    rate_score = RateScore(pred, gt)
    bleu_score = BleuScoreHF(pred, gt)
    bert_score = BERTScore(pred, gt)
    # green_score = GREENScore(pred, gt)
    rouge_score = ROUGEScore(pred, gt)
    meteor_score = METEORScore(pred, gt)

    results= [
        {
            "prediction": pred,
            "gt": gt,
            'rate_score': sum(rate_score) / len(rate_score), # returns the list of scores, we need to take the mean
            "bleu_score_hf": bleu_score['bleu'], # returns the avg,
            "bert_score": sum(bert_score['f1']) / len(bert_score['f1']), # returns the list of scores, we need to take the mean
            # "green_score": green_score,
            "rouge_score": rouge_score['rougeLsum'],
            "meteor_score": meteor_score['meteor']
        }
    ]

    eval_file = os.path.join(eval_dir, "eval_results_average.json")
    if eval_file is not None:
        # save evaluation as JSON
        if not Path(eval_file).parent.is_dir():
            os.makedirs(Path(eval_file).parent)
        with open(eval_file, 'w') as json_file:
            json.dump(results, json_file, indent=4)

def parse_args(default_config=None):
    if default_config is None:
        default_config = {}
    general_parser = argparse.ArgumentParser(description="Evaluation pipeline")
    general_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/vlm_eval",
        help="Directory to save evaluation outputs",
    )
    general_parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to inference results directory",
    )

    general_parser.add_argument(
        "--eval_config", type=str, help="Path to config file (YAML or JSON)"
    )
    # First parse just the config argument to load the file
    general_args, _ = general_parser.parse_known_args()

    # Load config file if provided
    if general_args.eval_config and os.path.exists(general_args.eval_config):
        file_ext = os.path.splitext(general_args.eval_config)[1].lower()
        with open(general_args.eval_config, "r") as file:
            if file_ext == ".yaml" or file_ext == ".yml":
                file_config = yaml.safe_load(file)
            elif file_ext == ".json":
                file_config = json.load(file)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")

            # Update config with values from file
            default_config.update(file_config)

    return general_args

def main():
    args = parse_args() # TODO: Add loggers
    # log_file = os.path.join(
    #     args.output_dir, datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_inference_{uuid4()}.log")
    # )
    # logger.add(log_file)

    # set_seed(seed)
    # get_scores_individual(inference_results_file=args.results_dir, eval_dir=args.output_dir)
    get_scores_avg(inference_results_file=args.results_dir, eval_dir=args.output_dir)


if __name__ == '__main__':
    main()
    # input_file_path = Path('./owl/results/inference_results_owl.json')
    # output_dir = Path('./owl/results/eval_results_owl.json')
    # with open(input_file_path, 'r') as json_file:
    #     input_data = json.load(json_file)

    # main(input_data, output_dir)