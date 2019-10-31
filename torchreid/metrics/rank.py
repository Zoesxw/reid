import numpy as np


def eval_rank(distmat, q_pids, g_pids):
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []

    for q_idx in range(num_q):
        cmc = matches[q_idx]
        all_cmc.append(cmc[0])

        n_cmc = cmc[:200]
        num_rel = n_cmc.sum()
        tmp_cmc = n_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * n_cmc
        if num_rel != 0:
            AP = tmp_cmc.sum() / num_rel
        else:
            AP = 0
        all_AP.append(AP)

    rank1 = np.mean(all_cmc)
    mAP = np.mean(all_AP)
    average = 0.5 * rank1 + 0.5 * mAP

    return rank1, mAP, average
