
# Matches a multiple-choice VQA option line like:
# "A. The cat is sleeping"
#
# Groups:
#   (1) [A-D]   -> the option letter (A, B, C, or D)
#   (2) (.+)    -> the option text itself (the answer content)
VQA_OPTION_PATTERN = r'^([A-D])\.\s*(.+)$'

# Matches an answer statement such as:
# "The answer is D."
# "The correct answer is B."
# "C"
#
# Groups:
#   (1) ((?:The (?:correct )?answer is )?) -> optional prefix ("The answer is " / "The correct answer is ")
#   (2) ([A-D])                            -> the answer letter (A–D)
#   (3) (\.?)                              -> optional period after the letter
VQA_ANSWER_PATTERN = r'((?:The (?:correct )?answer is )?)([A-D])(\.?)'
