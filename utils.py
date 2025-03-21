import transformers


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
