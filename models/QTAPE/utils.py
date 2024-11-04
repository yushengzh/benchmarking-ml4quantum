import torch
import numpy as np


def read_matrix_v2(matrix):
    # '[1, 2, 3]' ----> [1, 2, 3]
    matrix = matrix.replace("[", "").replace("]", "").split(",")
    return [float(x) for x in matrix]

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True