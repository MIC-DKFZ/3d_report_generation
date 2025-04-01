import transformers


def extract_raw_samples_from_batch(batch):
    return {
        "image_id": [sample["image_id"] for sample in batch],
        "question": [sample["question"] for sample in batch],
        "answer": [sample["full_answer"] for sample in batch],
        "answer_type": [sample["answer_type"] for sample in batch],
        "answer_subtype": [sample["answer_subtype"] for sample in batch],
    }


def extract_batch_prompt_only(inputs):
    prompt_only_key = [k for k in inputs.keys() if "prompt_only" in k]
    non_prompt_only_key = [k for k in inputs.keys() if "prompt_only" not in k]
    batch_out = {
        k: (
            inputs[k]
            if f"{k}_prompt_only" not in prompt_only_key
            else inputs[f"{k}_prompt_only"]
        )
        for k in non_prompt_only_key
    }
    # Wrap in BatchFeature to be able to move to gpu
    batch_out = transformers.BatchFeature(batch_out)
    return batch_out
