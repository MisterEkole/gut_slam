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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


''' Fxn to reconstruct a 3D point on a cylindrical mesh from its 2D image coordinates and the vanishing point.'''

def reconstruct_cylindrical_mesh(p, vp, p_0, R):
    # Calculate the direction of the cylinder's axis
    u = vp / np.linalg.norm(vp)

    # Construct an orthonormal basis (u, v, w)
    v = np.cross(u, p)
    v /= np.linalg.norm(v)

    w = np.cross(u, v)

    # Calculate the coefficients for u_d = a * v + b * w
    a = -np.dot(v, p_0) / R
    b = -np.sqrt(1 - a**2)

    # Calculate the unit vector u_d
    u_d = a * v + b * w

    # Calculate the angle theta between u_d and the x-axis of the cylindrical coordinate system
    y_cyl = np.cross(u, np.array([1, 0, 0]))
    y_cyl /= np.linalg.norm(y_cyl)
    x_cyl = np.cross(y_cyl, u)
    theta = np.arctan2(np.dot(u_d, y_cyl), np.dot(u_d, x_cyl))

    # Solve for h and z
    coefficients = np.linalg.lstsq(-np.column_stack((u, p)), (p_0 + R * u_d), rcond=None)[0]
    h, z = coefficients

    # Calculate 3D coordinates of the point P
    P = z * p

    return P, theta, h, u_d



''' Fxn to reconstruct a 3D point on a cylindrical geometry from its 2D image coordinates and the vanishing point.'''
def reconstruct_cylindrical_geometry(p, vp, p_0, h, theta):
    # Calculate the direction of the cylinder's axis
    u = vp / np.linalg.norm(vp)

    # Reconstruct a basis on the cylinder
    x_cyl = np.cross(u, -np.array([0, 1, 0]))
    x_cyl /= np.linalg.norm(x_cyl)
    y_cyl = np.cross(u, x_cyl)

    # Calculate the unit vector u_d in the (p_0, x_cyl, y_cyl, u) basis using theta
    u_d = np.cos(theta) * x_cyl + np.sin(theta) * y_cyl

    # Calculate the radial component R
    cross_product = np.cross(p, p_0) + h * np.cross(p, u)
    R = np.linalg.norm(cross_product) / np.linalg.norm(np.cross(p, u_d))

    # Solve for R and z
    coefficients = np.linalg.lstsq(-np.column_stack((u_d, p)), (p_0 + h * u), rcond=None)[0]
    R, z = coefficients

    # Calculate 3D coordinates of the point P
    P = z * p

    return P, R, u_d



''' Fxn to project 3D points on a cylindrical mesh to 2D image coordinates.'''

def project_cylindrical_mesh(theta, h, nappe, K, camTcyl, zmin):
    """
    Project 3D points on a cylindrical mesh to 2D image coordinates.

    Parameters:
    - theta: Azimuthal angles of the cylindrical mesh.
    - h: Heights of the cylindrical mesh.
    - nappe: Radial distances of the cylindrical mesh.
    - K: Camera intrinsic matrix.
    - camTcyl: Transformation matrix from cylindrical mesh coordinates to camera coordinates.
    - zmin: Minimum depth value for clipping.

    Returns:
    - pts2d: Projected 2D image coordinates.
    """

    nb_pts = len(theta)

    vec_theta = np.reshape(theta, (nb_pts, 1))
    vec_h = np.reshape(h, (nb_pts, 1))
    vec_nappe = np.reshape(nappe, (nb_pts, 1))

    pts3d = np.vstack([
        vec_nappe * np.cos(vec_theta),
        vec_nappe * np.sin(vec_theta),
        vec_h,
        np.ones((1, nb_pts))
    ])

    pts3d = np.dot(camTcyl, pts3d)
    ii = np.where(pts3d[2, :] < zmin)[0]
    vec_pts2d = np.dot(K, pts3d[:3, :] / (np.ones((3, 1)) * pts3d[2, :]))

    vec_pts2d[:, ii] = np.nan

    pts2d = np.reshape(vec_pts2d, (3 * len(theta), len(theta)))

    return pts2d



''' Fxn to create a cylindrical mesh.'''

def create_cylindrical_mesh(nb_pts_radiaux, nb_pts_axiaux, rayon, hauteur):
    """
    Create a cylindrical mesh.

    Parameters:
    - nb_pts_radiaux: Number of radial points.
    - nb_pts_axiaux: Number of axial points.
    - rayon: Radius of the cylindrical mesh.
    - hauteur: Height of the cylindrical mesh.

    Returns:
    - theta: Azimuthal angles of the cylindrical mesh.
    - h: Heights of the cylindrical mesh.
    - nappe: Radial distances of the cylindrical mesh.
    - pts3d: 3D coordinates of the points on the cylindrical mesh.
    """

    delta_theta = 2 * np.pi / nb_pts_radiaux
    delta_h = hauteur / (nb_pts_axiaux - 1)
    zmin=0.1

    # Generate an exponentially spaced grid for h
    lambda_val = 4
    expspace = np.exp(lambda_val * np.linspace(np.log(zmin) / lambda_val, np.log(hauteur) / lambda_val, nb_pts_axiaux))

    theta, h = np.meshgrid(np.linspace(0, 2 * np.pi - delta_theta, nb_pts_radiaux), expspace)
    

    #vec_theta = np.reshape(theta, nb_pts, 1)
    vec_theta=np.reshape(theta,(nb_pts_radiaux*nb_pts_axiaux,1))
    vec_h=np.reshape(h,(nb_pts_radiaux*nb_pts_axiaux,1))
    #vec_h = np.reshape(h, nb_pts, 1)

    nappe = rayon * np.ones_like(theta)

    pts3d = np.vstack([
        rayon * np.cos(vec_theta.T),
        rayon * np.sin(vec_theta.T),
        vec_h.T,
        np.ones_like(vec_h).T
    ])

    return theta, h, nappe, pts3d



''' Fxn to create a 3D model of the cylindrical mesh.'''

def create_I_model(pts2d, I_new, theta):
    pts2d = np.reshape(pts2d, (3 * len(theta), len(theta)))

    pts2d_x = np.array([pts2d[0 + 3 * (i - 1), :] for i in range(1, len(theta) + 1)])
    pts2d_y = np.array([pts2d[1 + 3 * (i - 1), :] for i in range(1, len(theta) + 1)])

    I_model_new = Poly3DCollection([], facecolors=[], edgecolors=[])
    k = 0

    for i in range(len(theta) - 1):
        for j in range(len(theta[0])):
            if j == len(theta[0]) - 1:
                vertices = np.array([
                    [pts2d_x[i, j], pts2d_x[i, j - 1], pts2d_x[i + 1, j - 1], pts2d_x[i + 1, j]],
                    [pts2d_y[i, j], pts2d_y[i, j - 1], pts2d_y[i + 1, j - 1], pts2d_y[i + 1, j]]
                ]).T
                I_model_new.add(Poly3DCollection([vertices], facecolors=[I_new[k, :]], edgecolors='none'))
                k += 1
                break

            vertices = np.array([
                [pts2d_x[i, j], pts2d_x[i, j + 1], pts2d_x[i + 1, j + 1], pts2d_x[i + 1, j]],
                [pts2d_y[i, j], pts2d_y[i, j + 1], pts2d_y[i + 1, j + 1], pts2d_y[i + 1, j]]
            ]).T
            I_model_new.add(Poly3DCollection([vertices], facecolors=[I_new[k, :]], edgecolors='none'))
            k += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.add_collection3d(I_model_new)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 360])
    ax.set_ylim([0, 360])
    ax.set_zlim([0, 360])

    plt.show()

    return I_model_new

''' Fxn to adjust a cylindrical mesh.'''

def adjust_cylindrical_mesh(theta, h, nappe, K, vp, p_0, R):
    nb_pts = len(theta)

    global zmin

   

    vec_theta = np.reshape(theta, (theta.size,1))
    vec_h = np.reshape(h, (h.size, 1))
    vec_nappe = np.reshape(nappe, (nappe.size, 1))

    #P_0 = np.dot(np.linalg.inv(K), p_0) * np.ones((1, nb_pts))
    P_0 = np.dot(np.linalg.inv(K), p_0.reshape((3, 1))) * np.ones((1, nb_pts))

    u = np.dot(np.linalg.inv(K), vp)
    u /= np.linalg.norm(u)
    #u_d = (np.eye(3) - np.outer(u[:3], u[:3])) @ np.vstack([np.cos(vec_theta), np.sin(vec_theta), np.zeros((1, nb_pts))])
    #u_d = (np.eye(3) - np.outer(u[:3], u[:3])) @ np.vstack([np.cos(vec_theta), np.sin(vec_theta), np.zeros((1, nb_pts))]).reshape((3, nb_pts))
    #u_d = (np.eye(3) - np.outer(u[:3], u[:3])) @ np.vstack([np.cos(vec_theta), np.sin(vec_theta), np.zeros((1, nb_pts))]).squeeze()
    #u_d = (np.eye(3) - np.outer(u[:3], u[:3])) @ np.concatenate([np.cos(vec_theta), np.sin(vec_theta), np.zeros((1, nb_pts))], axis=0)

    #u_d = (np.eye(3) - np.outer(u[:3], u[:3])) @ np.concatenate([np.cos(vec_theta), np.sin(vec_theta), np.zeros((1, nb_pts))], axis=0).reshape((3, nb_pts))
    u_d = (np.eye(3) - np.outer(u[:3], u[:3])) @ np.concatenate([np.cos(vec_theta), np.sin(vec_theta), np.zeros((1, nb_pts))], axis=0)






    u_d /= np.diag(np.sqrt(np.diag(u_d.T @ u_d)))

    pts3d = P_0 + R * u_d + np.outer(u, vec_h)

    camTcyl = np.vstack([u_d[:, 0], np.cross(u[:3], u_d[:, 0]), u[:3], P_0[:, 0]])
    camTcyl = np.vstack([camTcyl, [0, 0, 0, 1]])

    ii = np.where(pts3d[2, :] < zmin)
    vec_pts2d = K @ (pts3d[:3, :] / (np.ones((3, 1)) * pts3d[2, :]))
    vec_pts2d[:, ii] = np.nan

    pts2d = np.reshape(vec_pts2d, (3 * len(theta), len(theta)))

    return pts2d, pts3d, camTcyl



def display_cylindrical_mesh_proj(theta, h, nappe, K, camTcyl):
    '''
    camTcyl:camera to cylindrical mesh transformation matrix
    K: camera intrinsic matrix
    nappe: radial distances of the cylindrical mesh
    h: heights of the cylindrical mesh
    theta: azimuthal angles of the cylindrical mesh
    '''
    nb_pts = len(theta)

    fig_num = plt.gcf()
    hold_state = plt.ishold(fig_num)

    if len(theta) > 1:  # Normal case

        global zmin

        pts2d_x = []
        pts2d_y = []
        pts3d_list = []

        for i in range(len(theta)):

            pts3d = np.vstack([
                nappe[i, :] * np.cos(theta[i, :]),
                nappe[i, :] * np.sin(theta[i, :]),
                h[i, :],
                np.ones(len(theta[i, :]))
            ])

            pts3d = np.dot(camTcyl, pts3d)
            pts3d_list.append(pts3d)
            ii = np.where(pts3d[2, :] > zmin)
            pts2d = np.dot(K, pts3d[:3, ii]) / (np.ones((3, 1)) * pts3d[2, ii])

            pts2d_x.append(pts2d[0, :])
            pts2d_y.append(pts2d[1, :])

            plt.plot(pts2d[0, :], pts2d[1, :], 'r')

            if i == 0 and not hold_state:
                plt.hold(True)

        for j in range(len(theta[0])):
            pts3d = np.vstack([
                nappe[:, j] * np.cos(theta[:, j]),
                nappe[:, j] * np.sin(theta[:, j]),
                h[:, j],
                np.ones(len(theta[:, j]))
            ])

            pts3d = np.dot(camTcyl, pts3d)

            pts2d = np.dot(K, pts3d[:3, :]) / (np.ones((3, 1)) * pts3d[2, :])

            ii = np.where(pts3d[2, :] > zmin)
            plt.plot(pts2d[0, ii], pts2d[1, ii], 'b')

        for i in range(len(theta) - 1):
            for j in range(len(theta[0])):
                if j == len(theta[0]) - 1:
                    cross_u = np.array([pts2d_x[i + 1][j - j + 1] - pts2d_x[i][j - j + 1],
                                        pts2d_y[i + 1][j - j + 1] - pts2d_y[i][j - j + 1], 0])
                    cross_v = np.array([pts2d_x[i][j] - pts2d_x[i][j - j + 1],
                                        pts2d_y[i][j] - pts2d_y[i][j - j + 1], 0])
                    cross_prod = np.cross(cross_u, cross_v)

                    plt.fill([pts2d_x[i][j], pts2d_x[i][j - j + 1], pts2d_x[i + 1][j - j + 1], pts2d_x[i + 1][j]],
                             [pts2d_y[i][j], pts2d_y[i][j - j + 1], pts2d_y[i + 1][j - j + 1], pts2d_y[i + 1][j]],
                             color=(1, 0.77, 0), edgecolor=(0, 0, 0))

                    break

                cross_u = np.array([pts2d_x[i + 1][j + 1] - pts2d_x[i][j + 1],
                                    pts2d_y[i + 1][j + 1] - pts2d_y[i][j + 1], 0])
                cross_v = np.array([pts2d_x[i][j] - pts2d_x[i][j + 1],
                                    pts2d_y[i][j] - pts2d_y[i][j + 1], 0])
                cross_prod = np.cross(cross_u, cross_v)

                plt.fill([pts2d_x[i][j], pts2d_x[i][j + 1], pts2d_x[i + 1][j + 1], pts2d_x[i + 1][j]],
                         [pts2d_y[i][j], pts2d_y[i][j + 1], pts2d_y[i + 1][j + 1], pts2d_y[i + 1][j]],
                         color=(1, 0.77, 0), edgecolor=(0, 0, 0))

    else:  # If len(theta) <= 1, theta contains already calculated pts2d

        umax = 540
        vmax = 480
        tol = 0.1

        for i in range(1, len(theta) + 1):
            pts2d = theta[3 * i - 2:3 * i, :]
            plt.plot(pts2d[0, :], pts2d[1, :], 'r')
            if i == 1 and not hold_state:
                plt.hold(True)

        for j in range(len(theta[0])):
            pts2d = theta[:, j]
            plt.plot(pts2d[0::3], pts2d[1::3], 'b')

    plt.axis([0, umax, 0, vmax])

    if not hold_state:
        plt.hold(False)

    plt.show()

    return



''' Fxn to display a cylindrical mesh.'''

def display_cylindrical_mesh(theta, h, nappe, camTcyl=None):
    nb_pts = len(theta)

    fig_num = plt.gcf()
    hold_state = plt.ishold(fig_num)

    if camTcyl is None:
        camTcyl = np.eye(4)

   
    colors = ['r', 'g', 'b']
    for i in range(3):
        vector = np.column_stack([camTcyl[:3, 3], camTcyl[:3, 3] + camTcyl[:3, i]])
        plt.plot(vector[0, :], vector[1, :], vector[2, :], color=colors[i])
        if i == 0 and not hold_state:
            plt.hold(True)

    for i in range(len(theta)):

        pts3d = np.vstack([
            nappe[i, :] * np.cos(theta[i, :]),
            nappe[i, :] * np.sin(theta[i, :]),
            h[i, :],
            np.ones(len(theta[i, :]))
        ])

        pts3d = np.dot(camTcyl, pts3d)
        plt.plot(pts3d[0, :], pts3d[1, :], pts3d[2, :], 'r')

    for j in range(len(theta[0])):

        pts3d = np.vstack([
            nappe[:, j] * np.cos(theta[:, j]),
            nappe[:, j] * np.sin(theta[:, j]),
            h[:, j],
            np.ones(len(theta[:, j]))
        ])

        pts3d = np.dot(camTcyl, pts3d)
        plt.plot(pts3d[0, :], pts3d[1, :], pts3d[2, :], 'b')

    if not hold_state:
        plt.hold(False)

    plt.show()

    return

