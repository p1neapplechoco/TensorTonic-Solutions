import numpy as np


def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)

    if max_len is None:
        max_len = np.max([len(seq) for seq in seqs])

    res = np.full((N, max_len), pad_value)

    for i in range(N):
        L = min(max_len, len(seqs[i]))
        for j in range(L):
            res[i][j] = seqs[i][j]

    return res
