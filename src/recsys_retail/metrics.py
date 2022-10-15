import numpy as np


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)

    return precision