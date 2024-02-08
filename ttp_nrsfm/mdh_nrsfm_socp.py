'''
MDH_NRSFML Maximum Depth Heuristic with Non-Rigid Structure from Motion under the inextensibility constraint.
and the condition that each depth and each distance is non-negative.
'''

import cvxpy as cp
import numpy as np
from scipy.sparse import csc_matrix

def MDH_NrSfM(IDX, m, vis, max_depth_heuristic, solver='MOSEK'):
    M = len(m)
    N = m[0].shape[1]

    # Default values for visibility and solver
    if not vis:
        vis = [np.ones(N) for _ in range(M)]
    if solver not in ['MOSEK', 'SCS']:
        solver = 'MOSEK'

    # Flatten matrices
    m_flat = [mi.flatten() for mi in m]
    vis_flat = [vi.flatten() for vi in vis]

    # Indices of visible points
    P = np.where(np.concatenate(vis_flat))[0]

    # Depths for invisible points will be discarded
    nparams_mu2 = len(P)

    # Indices of distance variables
    P = np.concatenate([P, np.arange(N * M, N * M + IDX.size - len(IDX))])

    nparams2 = len(P)
    nparams = N * M + IDX.size - len(IDX)
    nparams_mu = N * M
    nparams_D = IDX.size - len(IDX)

    IDXt = IDX[:, 1:]
    IDXt2 = np.tile(IDX[:, 0], (1, IDXt.shape[1]))

    nconics = np.sum(np.logical_and(np.concatenate(vis_flat)[IDXt.flatten()], np.concatenate(vis_flat)[IDXt2.flatten()]))

    A = csc_matrix((nconics * 4, nparams), dtype=float)
    q = np.ones(nconics, dtype=int)

    ni = 0
    ni2 = 0

    for k in range(M):
        pp1 = np.where(np.logical_and(vis_flat[k][IDXt.flatten()], vis_flat[k][IDXt2.flatten()]))[0]
        nconicsk = len(pp1)
        idxi = IDXt2.flatten()[pp1]
        idxij = IDXt.flatten()[pp1]
        #idxj = np.arange(1, IDXt.shape[1])[pp1]
        idxj = np.arange(1, IDXt.shape[1] + 1)[pp1]
        idxk = np.full_like(idxj, k)
        idx1v = np.ravel_multi_index((idxi, idxk), (N, M))
        idx2v = np.ravel_multi_index((idxij, idxk), (N, M))
        idx3v = nparams_mu + np.ravel_multi_index((idxi, idxj), (N, IDXt.shape[1]))

        allA = np.arange(0, nconicsk * 4, 4)
        qq = np.concatenate([np.ones(nconicsk), m[k][0, idxi], m[k][1, idxi],
                             np.ones(nconicsk), -m[k][0, idxij], -m[k][1, idxij], -np.ones(nconicsk)])
        ind_conic = np.ravel_multi_index((np.repeat(allA, 7), np.tile(allA, 7)), (nconicsk * 4, nconicsk * 4))
        ind_param = np.concatenate([idx3v, idx1v, idx1v, idx1v, idx2v, idx2v, idx2v])
        Ak = csc_matrix((qq, (ind_conic, ind_param)), shape=(nconicsk * 4, nparams))
        A[ni:ni + nconicsk * 4, :] = Ak
        ni += nconicsk * 4
        ni2 += nconicsk

    A = A[:, P]

    # Compose F_struc matrix
    C = np.zeros(nparams2)
    C[:nparams_mu2] = -1

    A_eq = csc_matrix(np.concatenate([[np.zeros(nconics * 4)], A], axis=0))
    A_eq = csc_matrix(np.concatenate([[10, np.zeros(1), -np.ones(1)], A_eq], axis=0))

    # Define positive depth and distance variables
    variables = cp.Variable(nparams2)
    positive_depth_constraints = [variables[:nparams_mu2] >= 0]
    positive_distance_constraints = [variables[nparams_mu2:] >= 0]

    # Add inextensibility constraint with a maximum depth heuristic
    max_depth_constraints = [variables[:nparams_mu2] <= max_depth_heuristic]

    # Combine all constraints
    all_constraints = [A_eq @ variables == 0] + positive_depth_constraints + positive_distance_constraints + max_depth_constraints

    model = cp.Problem(cp.Maximize(C @ variables), all_constraints)
    model.solve(solver=solver, verbose=True)

    solu = np.array(variables.value)

    mu = np.zeros((M, N))
    mu[P[:nparams_mu2]] = solu[:nparams_mu2]
    mu = mu.T

    D = solu[nparams_mu2:nparams2].reshape(IDX.shape[0], IDX.shape[1] - 1)

    return mu, D

# Example usage:
# mu, D = MDH_NrSfM(IDX, m, vis, max_depth_heuristic, solver='MOSEK')
