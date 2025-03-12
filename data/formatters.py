from typing import Callable

_formatters = {}


# New formatters are registered with name <vlm>_<dataset>_<question_type>
def register_formatter(name: str) -> Callable:
    """Decorator to register a new formatter

    Args:
        name (str): name to register under

    Returns:
        registered callable
    """

    def _inner_wrapper(func: Callable) -> Callable:
        _formatters[name] = func
        return func

    return _inner_wrapper


def get_formatter(name: str):
    if name not in _formatters:
        raise ValueError(f"Formatter {name} not found")
    return _formatters[name]


@register_formatter("qwen_amos_qa")
def format_qwen_amos_qa_data(embedding_path, qa_pair):
    """Format QA pair for training"""
    options_text = "\n".join([f"{key}. {value}" for key, value in qa_pair["options"].items()])
    correct_answer_text = qa_pair["options"][qa_pair["answer"]]

    if "reasoning" not in qa_pair:
        qa_pair["reasoning"] = ""

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a medical imaging AI trained to answer questions about CT scans."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "embedding_path": embedding_path,
                },
                {
                    "type": "text",
                    "text": f"{qa_pair['question']}\n\n{options_text}"
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"The answer is {qa_pair['answer']}. {correct_answer_text}. {qa_pair['reasoning']}"
                }
            ],
        },
    ]


@register_formatter("qwen_amos_regions")
def format_qwen_amos_region_question(embedding_path, region, findings):
    """Create a simple question about a specific region"""
    if not findings:
        return None

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a medical imaging AI trained to analyze CT scans."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "embedding_path": embedding_path,
                },
                {
                    "type": "text",
                    "text": f"What do you see in the {region} region of this CT scan?"
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": findings
                }
            ],
        },
    ]


@register_formatter("qwen_lidc_qa")
def format_qwen_amos_qa_data(embedding_path, qa_pair):
    """Format QA pair for training"""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a medical imaging AI trained to answer questions about CT scans."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "embedding_path": embedding_path,
                },
                {
                    "type": "text",
                    "text": f"{qa_pair['question']}"
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{qa_pair['answer']}"
                }
            ],
        },
    ]