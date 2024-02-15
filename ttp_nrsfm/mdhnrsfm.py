import numpy as np
import cvxpy as cp

def NrSfM(IDX, m, vis, solver='MOSEK'):
    # Setting default values for visibility (1) and solver (mosek)
    if not vis:
        M = len(m)
        N = m[0].shape[1]
        vis = [np.ones(N) for _ in range(M)]

    M = len(m)
    N = m[0].shape[1]

    # Default parameters for CVXPY SOCP model.
    P = np.concatenate(vis).astype(bool)
    nparams_mu2 = len(P)

    P = np.concatenate([P, np.arange(N * M + 1, N * M + IDX.size - M + 1)])
    nparams2 = len(P)
    nparams = N * M + IDX.size - M
    nparams_mu = N * M
    nparams_D = IDX.size - M

    IDXt = IDX[:, 1:]
    IDXt2 = np.tile(IDX[:, 0], (1, IDXt.shape[1]))

    nconics = np.sum(np.logical_and(np.concatenate(vis)[IDXt.flatten()], np.concatenate(vis)[IDXt2.flatten()]))

    A = np.zeros((nconics * 4, nparams))
    q = np.ones(nconics)

    ni = 0
    ni2 = 0

    for k in range(M):
        pp1 = np.where(np.logical_and(np.concatenate(vis)[IDXt.flatten()], np.concatenate(vis)[IDXt2.flatten()]))[0]

        nconicsk = len(pp1)
        idxi = IDXt2.flatten()[pp1]
        idxij = IDXt.flatten()[pp1]
        idxj = np.arange(1, IDXt.shape[1])
        idxk = np.full_like(idxj, k)
        idx1v = np.ravel_multi_index((idxi, idxk), (N, M))
        idx2v = np.ravel_multi_index((idxij, idxk), (N, M))
        idx3v = nparams_mu + np.ravel_multi_index((idxi, idxj), (N, IDXt.shape[1]))

        allA = np.arange(0, nconicsk * 4, 4)
        qq = np.concatenate([np.ones(nconicsk), m[k][0, idxi], m[k][1, idxi],
                             np.ones(nconicsk), -m[k][0, idxij], -m[k][1, idxij], -np.ones(nconicsk)])

        ind_conic = np.concatenate([allA, allA + 1, allA + 2, allA + 3, allA + 1, allA + 2, allA + 3])
        ind_param = np.concatenate([idx3v, idx1v, idx1v, idx1v, idx2v, idx2v, idx2v])

        A[ni:ni + nconicsk * 4, :] = np.zeros((nconicsk * 4, nparams))
        A[ni:ni + nconicsk * 4, ind_param] = qq.reshape((4, nconicsk)).T

        ni += nconicsk * 4
        ni2 += nconicsk

    A = A[:, P]
    C = np.zeros(nparams2)
    C[:nparams_mu2] = -1

    A = np.concatenate([np.zeros((nconics * 4, 1)), A], axis=1)
    A = np.concatenate([np.array([[10] + [0] * (nparams2 - 1)]), A])

    x = cp.Variable(nparams2)
    constraints = [A @ x == 0]

    obj = cp.Maximize(C @ x)
    prob = cp.Problem(obj, constraints)

    if solver == 'MOSEK':
        prob.solve(solver=cp.MOSEK)
    elif solver == 'ECOS':
        prob.solve(solver=cp.ECOS)
    else:
        prob.solve(solver=cp.MOSEK)

    solu = x.value

    mu = np.zeros((M, N))
    mu[P[:nparams_mu2]] = solu[:nparams_mu2]
    mu = mu.T

    D = solu[nparams_mu2:nparams2]
    D = D.reshape((IDX.shape[0], IDX.shape[1] - 1))

    return mu, D
