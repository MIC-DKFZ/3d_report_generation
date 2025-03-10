import evaluate
from RaTEScore import RaTEScore

def BleuScoreHF(pred, gt, max_order=4):
    '''
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
    '''

    bleu = evaluate.load("bleu")
    b_score = bleu.compute(predictions=pred, references=gt, max_order=max_order)
    
    return b_score

def RateScore(pred, gt):
    '''
    pred ('list' of 'str'): Predicted text
        ex: pred = ["hello there general kenobi", "foo bar foobar"]
    gt ('list' of 'str'): Ground truth text
        ex: gt = ["hello there general kenobi", "foo bar foobar"]
    
    Note: 
            - RateScore can take a list of predictions and ground truth texts and calculate the score for each pair.
            - To calcualte the average score one needs to take the mean of the scores.
    '''
    ratescore = RaTEScore()
    score = ratescore.compute_score(pred, gt)

    return score

def BERTScore(pred, gt, model_type="distilbert-base-uncased"):
    '''
    predictions (list of str): Prediction/candidate sentences.
        ex: predictions = ["hello there general kenobi","foo bar foobar"]
    
    references (list of str or list of list of str): Reference sentences.
        ex multi reference per pred: references = [["hello there general kenobi","hello there general kenobi"],["foo bar foobar","foo bar foobar"]]
        ex single reference per pred: references = ["hello there general kenobi", "foo bar foobar","foo bar foobar"]

    lang (str): Language of the sentences; required (e.g. 'en').
    
    model_type (str): Bert specification, default using the suggested
        model for the target language; has to specify at least one of
        `model_type` or `lang`.
    '''
    bertscore = evaluate.load("bertscore")
    bert_score = bertscore.compute(predictions=pred, references=gt, model_type=model_type)
    
    return bert_score