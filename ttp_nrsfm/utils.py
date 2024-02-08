'''
getNeighborsVis Function
This function calculates a neighborhood matrix (IDX) based on pairwise city block distances
between normalized point correspondences in multiple views. The number of neighbors (Ng) is specified,
and visibility constraints (visb) are considered during the computation.

Inputs:
- m: List of numpy arrays representing normalized point correspondences for each view.
- Ng: Number of neighbors to consider.
- visb: Visibility constraints as a boolean matrix.

Output:
- IDX: Neighborhood matrix indicating indices of Ng nearest neighbors for each point.

register_to_ground_truth Function: This function registers a set of 3D points to a ground truth set of 3D points.

Author: Mitterand Ekole
Date: 08-02-2024
'''

import numpy as np
from scipy.spatial.distance import cdist

def getNeighborsVis(m, Ng, visb):
    N = m[0].shape[1]
    distmat = np.zeros((N, N, len(m)))

    for k in range(len(m)):
        distmat[:, :, k] = cdist(m[k].T, m[k].T, metric='cityblock')
        distmat[~visb[:, k], :, k] = -1

    dist = np.max(distmat, axis=2)
    

    IDX = np.argsort(dist, axis=1)[:, :Ng]

    return IDX


def register_to_ground_truth(Q, Qg):
    Qx = Q[0, :]
    Qy = Q[1, :]
    Qz = Q[2, :]
    px = Qg[0, :]
    py = Qg[1, :]
    pz = Qg[2, :]

    signo = 1
    alpha = np.linalg.inv(Qx @ Qx.T + Qy @ Qy.T + Qz @ Qz.T) @ (Qx @ px.T + Qy @ py.T + Qz @ pz.T)
    Qx = alpha * Qx
    Qy = alpha * Qy
    Qz = alpha * Qz

    error1 = np.sqrt((np.linalg.norm(pz - Qz) ** 2 + np.linalg.norm(px - Qx) ** 2 + np.linalg.norm(py - Qy) ** 2) / len(px))
    error2 = np.sqrt((np.linalg.norm(pz + Qz) ** 2 + np.linalg.norm(px + Qx) ** 2 + np.linalg.norm(py + Qy) ** 2) / len(px))

    if error2 < error1:
        signo = -1
        Qx = -Qx
        Qy = -Qy
        Qz = -Qz

    # Qx = Qx - np.mean(Qx)
    # Qy = Qy - np.mean(Qy)

    Q[0, :] = Qx
    Q[1, :] = Qy
    Q[2, :] = Qz

    return Q, alpha, signo

# Example usage:
# Q, alpha, signo = register_to_ground_truth(Q, Qg)


# Example usage:
# IDX = getNeighborsVis(m, Ng, visb)
