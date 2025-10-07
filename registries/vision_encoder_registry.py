from prep.evavit import PretrainedEvaEncFromEvaMAE


vision_encoder_registry: dict[str, type] = {
    "evamae": PretrainedEvaEncFromEvaMAE
}

