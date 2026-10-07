"""Timestamp and identifier helpers for run and log names."""

import random
import string
from datetime import datetime


def add_datetime_tag(s: str, sep: str = "__") -> str:
    return datetime.now().strftime(f"{s}{sep}%Y-%m-%d_%H-%M-%S")


def get_random_uid(length: int = 4) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
