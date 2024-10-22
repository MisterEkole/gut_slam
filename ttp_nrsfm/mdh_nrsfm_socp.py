'''
MDH_NRSFML Maximum Depth Heuristic with Non-Rigid Structure from Motion under the inextensibility constraint.
and the condition that each depth and each distance is non-negative.

Author: Mitterand Ekole
Date: 08-02-2024
'''

import cvxpy as cp
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
import json

def save_to_json(mu, D, file_path='output_data.json'):
    # Convert numpy arrays to lists for JSON serialization
    mu_list = mu.tolist()
    D_list = D.tolist()

    # Create a dictionary to store mu and D
    data = {
        'mu': mu_list,
        'D': D_list
    }

    # Save data to JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file)

    print(f'Data saved to {file_path}')

def MDH_NrSfM(IDX, m, vis, max_depth_heuristic, solver='ECOS'):
    M = len(m)
    N = m[0].shape[1]

    # Default values for visibility and solver
    # if not vis:
    #     vis = [np.ones(N) for _ in range(M)]
    # if solver not in ['MOSEK', 'ECOS', 'SCS']:
    #     solver = 'ECOS'

    # Flatten matrices
    m_flat = [mi.flatten() for mi in m]
    #vis_flat = [vis.flatten() for vi in vis]

    vis_flat=[np.ones(N) for _ in range(M)]

    # Indices of visible points
    P = np.where(np.concatenate(vis))[0]

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
        # if IDXt.shape[1] > 0:
        #     idxj = np.arange(IDXt.shape[1])[pp1] + 1
        # else:
        #     idxj = np.array([])
        if IDXt.size>0:
            idxj=pp1+1
        else:
            idxj=np.array([])
        idxk = np.full_like(idxj, k)
        idx1v = np.ravel_multi_index((idxi, idxk), (N, M))
        idx2v = np.ravel_multi_index((idxij, idxk), (N, M))
        idx3v = nparams_mu + np.ravel_multi_index((idxi, idxj), (N, N))

        allA = np.arange(0, nconicsk * 4, 4)
        qq = np.concatenate([np.ones(nconicsk), m[k][0, idxi], m[k][1, idxi],
        np.ones(nconicsk), -m[k][0, idxij], -m[k][1, idxij], -np.ones(nconicsk)])  #1D array containing the values of the non-zero elements of the sparse matrix
        ind_conic = np.ravel_multi_index((np.repeat(allA, 7), np.tile(allA, 7)), (nconicsk * 4, nconicsk * 4))
        ind_param = np.concatenate([idx3v, idx1v, idx1v, idx1v, idx2v, idx2v, idx2v])

        # Ensure that row and column indices are within the valid range
        valid_indices = np.logical_and(np.logical_and(ind_conic >= 0, ind_conic < nconicsk * 4), np.logical_and(ind_param >= 0, ind_param < nparams))
        ind_conic = ind_conic[valid_indices]
        ind_param = ind_param[valid_indices]
        qq = qq[valid_indices]



        Ak = csc_matrix((qq, (ind_conic, ind_param)), shape=(nconicsk * 4, nparams), dtype=float)  # sparse matrix rep conic constraints
        #A[ni:ni + nconicsk * 4, :] = Ak
        A[ni:ni + nconicsk * 4, P] = Ak[:, :]

      

    #A = A[:, P]

    # Compose F_struc matrix
    C = np.zeros(nparams2)
    C[:nparams_mu2] = -1


    A_eq = csc_matrix(np.concatenate([np.zeros((1, nparams2)), A.toarray()], axis=0))
    A_eq[0,0]=10 #set the first element of the first row to 10
 

    # Define positive depth and distance variables
    variables = cp.Variable(nparams2)
    positive_depth_constraints = [variables[:nparams_mu2] >= 0]
    positive_distance_constraints = [variables[nparams_mu2:] >= 0]

    # Add inextensibility constraint with a maximum depth heuristic
    max_depth_constraints = [variables[:nparams_mu2] <= max_depth_heuristic]

    # Combine all constraints
    all_constraints = [A_eq @ variables == 0] + positive_depth_constraints + positive_distance_constraints + max_depth_constraints

    model = cp.Problem(cp.Maximize(C @ variables), all_constraints)
    model.solve(solver='ECOS', verbose=True)

    mu=variables.value[:nparams_mu2]
    D=variables.value[nparams_mu2:nparams2].reshape(IDX.shape[0], IDX.shape[1] - 1)

    results={'mu':mu.tolist(),'D':D.tolist()}
    with open('output_data.json','w') as file:
        json.dump(results,file)
    
    ''' 
        ~to be removed
        #D = solu[nparams_mu2:nparams2].reshape(IDX.shape[0], IDX.shape[1] - 1)
        #D=variables.value[nparams_mu2:nparams2].reshape(IDX.shape[0], IDX.shape[1] - 1)
        #return variables.value[:nparams_mu2], variables.value[nparams_mu2:nparams2].reshape(IDX.shape[0], IDX.shape[1] - 1)
    '''
    return mu, D



