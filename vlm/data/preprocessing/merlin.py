import re


def clean_text(text):
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +([.;])", r"\1", text)
    return text
