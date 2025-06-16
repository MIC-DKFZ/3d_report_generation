import evaluate
from RaTEScore import RaTEScore


def BleuScoreHF(pred, gt, max_order=4):
    """
    FORMULA: BLEU = brevity_penalty * exp(sum(w_n * log(p_n)))

    predictions (`list` of `str`s): Translations to score.
        ex: predictions = ["hello there general kenobi","foo bar foobar"]
    references (`list` of `list`s of `str`s): references for each translation. If there is only one reference it can be list of strings as well
        ex multi reference per pred: references = [["hello there general kenobi","hello there general kenobi"],["foo bar foobar","foo bar foobar"]]
        ex single reference per pred: references = ["hello there general kenobi", "foo bar foobar","foo bar foobar"]
    max_order (`int`): Maximum n-gram order to use when computing BLEU score. Defaults to `4`.

        Note:
            - Bleu score can take a list of predictions and ground truth texts but oit calculates the average score over the dataset.
            - To calcualte the score per single pair one needs to pass only the pred and gt values of that single pair.
    """

    bleu = evaluate.load("bleu")
    b_score = bleu.compute(predictions=pred, references=gt, max_order=max_order)

    return b_score


def RateScore(pred, gt):
    """
    pred ('list' of 'str'): Predicted text
        ex: pred = ["hello there general kenobi", "foo bar foobar"]
    gt ('list' of 'str'): Ground truth text
        ex: gt = ["hello there general kenobi", "foo bar foobar"]

    Note:
            - RateScore can take a list of predictions and ground truth texts and calculate the score for each pair.
            - To calcualte the average score one needs to take the mean of the scores.
    """
    ratescore = RaTEScore()
    score = ratescore.compute_score(pred, gt)

    return score


def BERTScore(pred, gt, model_type="distilbert-base-uncased"):
    """
    predictions (list of str): Prediction/candidate sentences.
        ex: predictions = ["hello there general kenobi","foo bar foobar"]

    references (list of str or list of list of str): Reference sentences.
        ex multi reference per pred: references = [["hello there general kenobi","hello there general kenobi"],["foo bar foobar","foo bar foobar"]]
        ex single reference per pred: references = ["hello there general kenobi", "foo bar foobar","foo bar foobar"]

    lang (str): Language of the sentences; required (e.g. 'en').

    model_type (str): Bert specification, default using the suggested
        model for the target language; has to specify at least one of
        `model_type` or `lang`.
    """
    bertscore = evaluate.load("bertscore")
    bert_score = bertscore.compute(predictions=pred, references=gt, model_type=model_type)

    return bert_score


def GREENScore(pred, gt, model_name="StanfordAIMI/GREEN-RadLlama2-7b", output_dir="."):

    from green_score import GREEN

    """
    predictions (list of str): Prediction/candidate sentences.
        ex: predictions = ["hello there general kenobi","foo bar foobar"]

    references (list of str or list of list of str): Reference sentences.
        ex multi reference per pred: references = [["hello there general kenobi","hello there general kenobi"],["foo bar foobar","foo bar foobar"]]
        ex single reference per pred: references = ["hello there general kenobi", "foo bar foobar","foo bar foobar"]

    model_name (str): Pretrained model name from the GREEN model zoo (cf. https://github.com/Stanford-AIMI/GREEN?tab=readme-ov-file#benchmark).
    """
    supported_models = [
        "StanfordAIMI/GREEN-RadLlama2-7b",
        "StanfordAIMI/GREEN-RadPhi2",
        "StanfordAIMI/GREEN-Phi2",
        "StanfordAIMI/GREEN-Mistral-7b",
        "StanfordAIMI/GREEN-Llama2-7b",
        "StanfordAIMI/GREEN-Phi-3",
        "StanfordAIMI/GREEN-Gemma-2b-MM",
    ]
    assert (
        model_name in supported_models
    ), f"Model {model_name} not supported. Supported models are: {supported_models}"
    global green_scorer_fct
    if "green_scorer_fct" not in globals():
        green_scorer_fct = GREEN(model_name, output_dir=".")
    mean, std, green_score_list, summary, result_df = green_scorer_fct(gt, pred)
    return mean


def ROUGEScore(pred, gt):
    """
    predictions (list of str): Prediction/candidate sentences.
        ex: predictions = ["hello there general kenobi","foo bar foobar"]

    references (list of str or list of list of str): Reference sentences.
        ex multi reference per pred: references = [["hello there general kenobi","hello there general kenobi"],["foo bar foobar","foo bar foobar"]]
        ex single reference per pred: references = ["hello there general kenobi", "foo bar foobar","foo bar foobar"]

    tokenizer (str): Tokenizer to use. If None, defaults to the one provided by the model.
    """
    rouge = evaluate.load("rouge")
    r_score = rouge.compute(predictions=pred, references=gt, tokenizer=None)

    return r_score


def METEORScore(pred, gt, alpha=0.9):
    """
    predictions (list of str): Prediction/candidate sentences.
        ex: predictions = ["hello there general kenobi","foo bar foobar"]

    references (list of str or list of list of str): Reference sentences.
        ex multi reference per pred: references = [["hello there general kenobi","hello there general kenobi"],["foo bar foobar","foo bar foobar"]]
        ex single reference per pred: references = ["hello there general kenobi", "foo bar foobar","foo bar foobar"]

    alpha (float): Parameter for controlling relative weights of precision and recall. The default value is 0.9.
    beta (float): Parameter for controlling shape of penalty as a function of fragmentation. The default value is 3.
    gamma (float): The relative weight assigned to fragmentation penalty. The default is 0.5.
    """
    meteor = evaluate.load("meteor")
    m_score = meteor.compute(predictions=pred, references=gt, alpha=alpha)

    return m_score
