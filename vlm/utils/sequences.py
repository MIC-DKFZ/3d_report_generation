"""Sequence searching and balanced splitting."""


def find_sublist(lis: list, sub: list):
    m = len(sub)
    for i in range(len(lis) - m + 1):
        if lis[i : i + m] == sub:
            return i
    return -1


def chunkify(lis: list, num_chunks: int):
    """Split list into 'num_chunks' roughly equal-sized chunks."""
    k, m = divmod(len(lis), num_chunks)
    return [lis[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]
