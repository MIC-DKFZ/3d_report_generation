from typing import Callable, Dict, List, Optional, Tuple, Any

_formatters = {}


# New formatters are registered with name <vlm>_<dataset>_<question_type>
def register_formatter(name: str) -> Callable:
    """Decorator to register a new formatter"""

    def _inner_wrapper(func: Callable) -> Callable:
        _formatters[name] = func
        return func

    return _inner_wrapper


def get_formatter(name: str):
    if name not in _formatters:
        raise ValueError(f"Formatter {name} not found")
    return _formatters[name]


# Helper functions to build conversation parts
def build_system_message(text: str) -> Dict:
    """Build a system message with given text"""
    return {"role": "system", "content": [{"type": "text", "text": text}]}


def build_user_message_qwen(text: str, embedding_path: str) -> Dict:
    """Build a Qwen user message with image embedding and text"""
    return {
        "role": "user",
        "content": [{"type": "image", "embedding_path": embedding_path}, {"type": "text", "text": text}],
    }


def build_user_message_owl(text: str) -> Dict:
    """Build an Owl user message with <|image|> tag and text"""
    return {"role": "user", "content": [{"type": "text", "text": f"<|image|>\n{text}"}]}


def build_assistant_message(text: str) -> Dict:
    """Build an assistant message with given text"""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def create_qa_conversations(
    system_text: str, user_message: Dict, full_answer: str, minimal_answer: str = "The answer is"
) -> Tuple[List[Dict], List[Dict]]:
    """Create full and minimal conversations from components"""
    system_message = build_system_message(system_text)

    full_conversation = [system_message, user_message, build_assistant_message(full_answer)]

    minimal_conversation = [system_message, user_message]

    return full_conversation, minimal_conversation


@register_formatter("qwen_amos_qa")
def format_qwen_amos_qa_data(qa_data, embedding_path):
    """Format QA pair for training using extracted data with pre-generated texts"""
    # Build user message (Qwen style)
    user_message = build_user_message_qwen(qa_data["user_text"], embedding_path)

    return create_qa_conversations(
        qa_data["system_text"], user_message, qa_data["full_answer"], qa_data["minimal_answer"]
    )


@register_formatter("owl_amos_qa")
def format_owl_amos_qa_data(qa_data, embedding_path=None):
    """Format QA pair for training using extracted data with pre-generated texts"""
    # Build user message (Owl style)
    user_message = build_user_message_owl(qa_data["user_text"])

    return create_qa_conversations(
        qa_data["system_text"], user_message, qa_data["full_answer"], qa_data["minimal_answer"]
    )


@register_formatter("qwen_amos_regions")
def format_qwen_amos_region_question(region_data, embedding_path):
    """Format region question using pre-generated texts"""
    if not region_data:
        return None, None

    # Build user message (Qwen style)
    user_message = build_user_message_qwen(region_data["user_text"], embedding_path)

    return create_qa_conversations(
        region_data["system_text"], user_message, region_data["full_answer"], region_data["minimal_answer"]
    )


@register_formatter("owl_amos_regions")
def format_owl_amos_region_question(region_data, embedding_path=None):
    """Format region question using pre-generated texts"""
    if not region_data:
        return None, None

    # Build user message (Owl style)
    user_message = build_user_message_owl(region_data["user_text"])

    return create_qa_conversations(
        region_data["system_text"], user_message, region_data["full_answer"], region_data["minimal_answer"]
    )
