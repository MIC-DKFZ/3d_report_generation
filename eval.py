import argparse

from datetime import datetime
import os
import json
from pathlib import Path
from uuid import uuid4

from utils import logger
import yaml
from tqdm import tqdm

from Eval.metrics import BERTScore, RateScore, BleuScoreHF, ROUGEScore, METEORScore #, GREENScore

def get_scores(inference_results_file, eval_dir):
    '''
    This function loads the inference results from a JSON file and calculates various evaluation metrics for each sample and their average.
    The metrics include RateScore, BLEU score, BERT score, ROUGE score, and METEOR score.
    The results are saved in a JSON file in the specified directory.

    Note:
        For many scores we could also use the evaluate library directly to calculate the average scores by sending the complete GT and Pred lists.
        However, this approach calculates the average score by appending all predictions and all references into two long documents and calculating the metric on these two long documents.
        Since this average score is slightly different from the average of the individual scores (~ the third decimal place), we decided to calculate the average by averaging sample-level scores.
    
    Args:
        inference_results_file (str): Path to the JSON file containing inference results.
        eval_dir (str): Directory to save evaluation results.
    '''
    logger.info("Running evaluation...")

    # Load inference results from JSON file
    input_data = []
    with open(inference_results_file, 'r') as json_file: # TODO: maybe support other formats later
        input_data = json.load(json_file)

    results = []
    average_results = {
        "average_rate_score" : 0.0,
        "average_bleu_score" : 0.0,
        "average_bert_score" : 0.0,
        "average_rouge_score" : 0.0,
        "average_meteor_score" : 0.0,
    }
    for sample in tqdm(input_data):
        pred = sample['output_text']
        gt = sample['answer']

        rate_score = RateScore([pred], [gt])
        bleu_score = BleuScoreHF([pred], [gt])
        bert_score = BERTScore([pred], [gt])
        # green_score = GREENScore([pred], [gt])
        rouge_score = ROUGEScore([pred], [gt])
        meteor_score = METEORScore([pred], [gt])
        results.append(
            {   
                'image_id': sample['image_id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'answer_type': sample['answer_type'],
                'answer_subtype': sample['answer_subtype'],
                'output_text': sample['output_text'],
                'rate_score': rate_score[0],
                "bleu_score_hf": bleu_score['bleu'],
                "bert_score": bert_score['f1'][0],
                # "green_score": green_score,
                "rouge_score": rouge_score['rougeLsum'],
                "meteor_score": meteor_score['meteor']
            }
        )
        # accumulate average results
        average_results['average_rate_score'] += rate_score[0]
        average_results['average_bleu_score'] += bleu_score['bleu']
        average_results['average_bert_score'] += bert_score['f1'][0]
        average_results['average_rouge_score'] += rouge_score['rougeLsum']
        average_results['average_meteor_score'] += meteor_score['meteor']

    # calculate average scores
    average_results['average_rate_score'] /= len(input_data)
    average_results['average_bleu_score'] /= len(input_data)
    average_results['average_bert_score'] /= len(input_data)
    average_results['average_rouge_score'] /= len(input_data)
    average_results['average_meteor_score'] /= len(input_data)
    
    # add the average results to the top of the results list
    results.insert(0, average_results)
    logger.info("All scores are calculated.")

    eval_file = os.path.join(eval_dir, "eval_results.json")
    if eval_file is not None:
        logger.info(f"Saving evaluation results to {eval_file}")
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
    log_file = os.path.join(
        args.output_dir, datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_evaluation_{uuid4()}.log")
    )
    logger.add(log_file)

    get_scores(inference_results_file=args.results_dir, eval_dir=args.output_dir)


if __name__ == '__main__':
    main()