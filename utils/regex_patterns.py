
# Matches a multiple-choice VQA option line like:
# "A. The cat is sleeping"
#
# Groups:
#   (1) [A-D]   -> the option letter (A, B, C, or D)
#   (2) (.+)    -> the option text itself (the answer content)
VQA_OPTION_PATTERN = r'^([A-D])\.\s*(.+)$'


# Matches an answer reference in a VQA-style response, such as:
# "The answer is B." or simply "B." at the beginning of the string.
#
# The pattern enforces that the answer letter [A-D] appears either:
#   - at the very start of the string, OR
#   - immediately after "The answer is " or "The correct answer is ".
#
# This allows capturing both the optional prefix phrase and the
# predicted answer letter, while ignoring cases where letters
# appear naturally within sentences (e.g., "Option A shows...").
#
# Groups:
#   (1) (The (?:correct )?answer is )?  -> optional prefix text ("The answer is " or "The correct answer is ")
#   (2) [A-D]                           -> the predicted answer letter (A, B, C, or D)
#   (3) (\.?)                           -> optional trailing period after the letter
#
# Example matches:
#   "B."                        -> group(1)=None, group(2)="B", group(3)="."
#   "The answer is C."          -> group(1)="The answer is ", group(2)="C", group(3)="."
#   "The correct answer is D"   -> group(1)="The correct answer is ", group(2)="D", group(3)=""
VQA_ANSWER_PATTERN = r'(?:(The (?:correct )?answer is )|^)([A-D])(\.?)(?!\w)'


