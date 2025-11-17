
# Matches a multiple-choice VQA option line like:
# "A. The cat is sleeping"
#
# Groups:
#   (1) [A-D]   -> the option letter (A, B, C, or D)
#   (2) (.+)    -> the option text itself (the answer content)
VQA_OPTION_PATTERN = r'^([A-D])\.\s*(.+)$'


# Matches an answer reference in a VQA-style response, such as:
# "The answer is B.", "Answer: C", or simply "B." at the beginning of the string.
#
# The pattern allows the answer letter [A-D] to appear either:
#   - at the very start of the string, OR
#   - immediately after the phrases "The answer is " or "Answer:" (with optional whitespace).
#
# This captures both the optional prefix and the predicted answer letter,
# while preventing false matches where letters appear inside normal text
# (e.g., "Option A shows..." will not match).
#
# Groups:
#   (1) (The answer is |Answer:\s*)   -> optional prefix text ("The answer is " or "Answer:" + optional spaces)
#   (2) [A-D]                         -> the predicted answer letter (A, B, C, or D)
#   (3) (\.?)                         -> optional trailing period after the letter
#
# Lookahead:
#   (?!\w)                            -> ensures the match is not followed by a word character
#
# Example matches:
#   "B."                -> group(1)=None, group(2)="B", group(3)="."
#   "The answer is C."  -> group(1)="The answer is ", group(2)="C", group(3)="."
#   "Answer: D"         -> group(1)="Answer:", group(2)="D", group(3)=""
#
VQA_ANSWER_PATTERN = r'(?:(The answer is |Answer:\s*)|^)([A-D])(\.?)(?!\w)'


# Regex pattern to detect multiple-choice option letters ("A", "B", "C", "D") in natural text.
#
# Matches:
#     - Standalone "A" when it is not acting as the article before a noun
#       (i.e., not followed by another word and not appearing right after punctuation like '.', '(' or "'").
#     - Standalone "B", "C", or "D" bounded by word boundaries.
VQA_OPTION_LETTERS_IN_TEXT_PATTERN = r"(?:(?<![.(']\s)(?<![.('])\b(A)\b|\b(A)\b(?!\s*\w))|\b([BCD])\b"


