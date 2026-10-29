import re

import torch


IGNORE_LABEL_ID = -100
THINKING_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
THINKING_TAG_PATTERN = re.compile(r"</?think>")


def decode_label_answers(
    tokenizer,
    labels: torch.Tensor,
    ignore_label_id: int = IGNORE_LABEL_ID,
    skip_special_tokens: bool = True,
) -> list[str]:
    answers = []
    for label_row in labels:
        answer_ids = label_row[label_row != ignore_label_id].detach().cpu().tolist()
        answers.append(
            tokenizer.decode(
                answer_ids,
                skip_special_tokens=skip_special_tokens,
            ).strip()
        )
    return answers


def decode_generated_answers(
    tokenizer,
    outputs: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    skip_special_tokens: bool = True,
) -> list[str]:
    prompt_length = prompt_input_ids.shape[1]
    generated_ids = [
        output[prompt_length:].detach().cpu().tolist()
        for output in outputs
    ]
    return [
        answer.strip()
        for answer in tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )
    ]


def decode_prompts(
    tokenizer,
    input_ids: torch.Tensor,
    skip_special_tokens: bool = True,
) -> list[str]:
    return [
        prompt.strip()
        for prompt in tokenizer.batch_decode(
            input_ids.detach().cpu().tolist(),
            skip_special_tokens=skip_special_tokens,
        )
    ]


def has_thinking_tags(text: str) -> bool:
    return THINKING_TAG_PATTERN.search(text) is not None


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks and stray thinking tags from a generated answer."""
    text = THINKING_BLOCK_PATTERN.sub("", text)
    return THINKING_TAG_PATTERN.sub("", text).strip()
