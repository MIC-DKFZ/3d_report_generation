from green_score import GREEN

def eval_green_score(preds: list[str], gts: list[str], batch_size: int = 4) -> dict[str, list]:
    # 'output_dir' arg is not used at all
    green_scorer = GREEN(model_name="StanfordAIMI/GREEN-radllama2-7b")
    green_scorer.batch_size = batch_size
    mean, std, green_score_list, summary, result_df = green_scorer(preds, gts)

    return {
        "green_score": mean.item(),
        "green_score_std": std.item()
    }