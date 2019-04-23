#!/usr/bin/evn python

"""
Utility functions for SfM
"""

# Imports
import numpy as np
import cv2
import argparse
import pdb
import glob
from scipy.optimize import least_squares

def parsePoints(imnum_1, imnum_2, txtpath):
    # Parse the provided text files to extract the corresponding points
    textfilename = txtpath+str(imnum_1)+'.txt'
    points_mat = np.zeros((1,4))
    with open(textfilename) as f:
        for idx,line in enumerate(f):
            if idx==0:
                n_features = int(line.strip().split(' ')[1])
            else:
                n_matches = int(line.strip().split(' ')[0])-1
                (ui,vi) = line.strip().split(' ')[4:6]
                ui = float(ui)
                vi = float(vi)
                corr_points = line.strip().split(' ')[6:]
                for i in range(n_matches):
                    corr_img = int(corr_points[3*i])
                    ui_d = float(corr_points[3*i+1])
                    vi_d = float(corr_points[3*i+2])
                    if corr_img == imnum_2:
                        points_mat = np.append(points_mat, [[ui, vi, ui_d, vi_d]], axis=0)

    points_mat = points_mat[1:,:]

    return points_mat


def EstimateFundamentalMatrix(points_mat):

    # Generate the matrix A
    A_mat = GenerateMatrixA(points_mat)

    # Solve for Ax=0 by SVD
    umat, smat, vmat = np.linalg.svd(A_mat)
    soln = vmat.T[:,8]
    F = np.reshape(soln, (3,3)).T

    return F

def GetInlierRANSAC(points_mat):
    n=0
    n_iters = 500
    theta = 0.0025
    S_in = []
    for i in range(n_iters):
        # Select 8 correspondences at random from the set of points
        sample_points = points_mat[np.random.choice(points_mat.shape[0], 8, replace=False), :]
        # Solve for F using sample_points
        A_sample = GenerateMatrixA(sample_points)
        F_sample = EstimateFundamentalMatrix(sample_points)
        S = []
        # res_val = np.linalg.norm(np.matmul(A_sample, np.reshape(F_sample.T,(9,))), ord=1)
        for j in range(points_mat.shape[0]):
            X_i = np.array([[points_mat[j,0], points_mat[j,1], 1]])
            X_d = np.array([[points_mat[j,2], points_mat[j,3], 1]])

            res_val = np.matmul(np.matmul(X_d, F_sample), X_i.T)[0][0]

            if abs(res_val)<theta:
                S.append(j)
        if n<len(S):
            n = len(S)
            S_in = S

    # S_in gives the set of points with outliers removed using RANSAC
    points_RANSAC = points_mat[S_in, :]
    return points_RANSAC

def EssentialMatrixFromFundamentalMatrix(F_RANSAC, K):
    E = np.matmul(np.matmul(K.T, F_RANSAC), K)

    # Reconstruct E using the SVD decomposition in problem statement
    umat, smat, vmat = np.linalg.svd(E)

    E_sing = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])

    E_recon = np.matmul(np.matmul(umat, E_sing), vmat)
    return E_recon

def GenerateMatrixA(points_mat):
    A_mat = np.zeros((1,9))

    for i in range(points_mat.shape[0]):
        xi = points_mat[i][0]
        yi = points_mat[i][1]
        xi_d = points_mat[i][2]
        yi_d = points_mat[i][3]
        A_mat = np.append(A_mat, [[xi*xi_d, xi*yi_d, xi, yi*xi_d, yi*yi_d, yi, xi_d, yi_d, 1]], axis=0)

    A_mat = A_mat[1:,:]

    return A_mat

def ExtractCameraPose(K,C,R):
    # P = KR[I -C]
    if np.linalg.det(R)<1:
        C = -C
        R = -R
    temp = np.hstack((np.identity(3), -C.reshape((3,1))))
    P = np.matmul(np.matmul(K, R), temp)
    return P

def LinearTriangulation(P1, P2, points_mat):

    # Return 3D points extracted through linear triangulation
    X_soln = np.zeros((points_mat.shape[0],3))

    # Generate matrix to solve
    for i in range(points_mat.shape[0]):
        row1 = np.array([ points_mat[i][1]*P1[2][0] - P1[1][0], points_mat[i][1]*P1[2][1] - P1[1][1], \
                          points_mat[i][1]*P1[2][2] - P1[1][2], points_mat[i][1]*P1[2][3] - P1[1][3] ])

        row2 = np.array([ points_mat[i][0]*P1[2][0] - P1[0][0], points_mat[i][0]*P1[2][1] - P1[0][1], \
                          points_mat[i][0]*P1[2][2] - P1[0][2], points_mat[i][0]*P1[2][3] - P1[0][3] ])

        row3 = np.array([ points_mat[i][3]*P2[2][0] - P2[1][0], points_mat[i][3]*P2[2][1] - P2[1][1], \
                          points_mat[i][3]*P2[2][2] - P2[1][2], points_mat[i][3]*P2[2][3] - P2[1][3] ])

        row4 = np.array([ points_mat[i][2]*P2[2][0] - P2[0][0], points_mat[i][2]*P2[2][1] - P2[0][1], \
                          points_mat[i][2]*P2[2][2] - P2[0][2], points_mat[i][2]*P2[2][3] - P2[0][3] ])

        A_mat = np.vstack((row1, row2, row3, row4))

        # X_soln[i,:] = np.linalg.lstsq(A_mat, np.zeros((4,1)))[0].T
        # umat, smat, vh = np.linalg.svd(A_mat) # Solve using SVD
        # X_soln[i,:] = vh.T[:,-1]

        # We know X = [x y z 1].T, so solve the overdetermined system
        b = -A_mat[:,3]
        X_soln[i,:] = np.linalg.lstsq(A_mat[:,:3], b, rcond=None)[0]

    return X_soln

def DisambiguateCameraPose(Cset, Rset, Xset):
    # Identify correct camera using cheirality condition
    num_d = np.zeros(4)
    for idx in range(len(Cset)):
        C = Cset[idx]
        R = Rset[idx]
        X = Xset[idx]

        # Enforce condition [0 0 1]*X>0
        X = X[X[:,2]>0]

        # num_d[idx] = sum(p > 0 for p in np.matmul((Rset[idx])[2,:], (Xset[idx]-Cset[idx]).T))
        num_d[idx] = sum(p > 0 for p in np.matmul(R[2,:], (X-C).T))

    return Cset[np.argmax(num_d)], Rset[np.argmax(num_d)], Xset[np.argmax(num_d)]

def DisambiguateCameraPose_2Check(Cset, Rset, Xset, K, points_RANSAC):
    # Identify correct camera using cheirality condition
    num_d = np.zeros(4)
    res_d = np.zeros(4)
    P0 = ExtractCameraPose(K,np.zeros(Cset[0].shape), np.identity(Rset[0].shape[0]))

    for idx in range(len(Cset)):
        C = Cset[idx]
        R = Rset[idx]
        X = Xset[idx]
        P = ExtractCameraPose(K,C,R)

        # Enforce condition [0 0 1]*X>0
        # points_mat = points_RANSAC[X[:,2]>0]
        # X = X[X[:,2]>0]
        # if len(np.where(X[:,2]>0)[0])/len(X) < 0.5:
        #     num_d[idx] = 0
        #     res_d[idx] = 100000
        #     continue

        points_mat = points_RANSAC

        # num_d[idx] = sum(p > 0 for p in np.matmul((Rset[idx])[2,:], (Xset[idx]-Cset[idx]).T))
        num_d[idx] = sum(p > 0 for p in np.matmul(R[2,:], (X-C).T))

        residual = 0
        for i in range(X.shape[0]):
            # X_i = X[i]
            lossval = optim_func(X[i], P0, P, points_mat[i])
            #res_lsq = least_squares(optim_func, X0_flat, args=(P0, P, points_RANSAC[i]), max_nfev=max_nfev)
            #X0_nloptim[i,:] = res_lsq.x
            residual+=lossval
        residual /= X.shape[0]
        res_d[idx] = residual

    id1 = np.argsort(num_d)[-1]
    id2 = np.argsort(num_d)[-2]
    idxsel = id1

    #if (res_d[id1]>res_d[id2]) and ((num_d[id1] - num_d[id2])/num_d[id1] < 0.35):
    #    idxsel = id2

    if res_d[id1]>res_d[id2]:
        idxsel = id2

    # idxsel = np.argmin(res_d)

    return Cset[idxsel], Rset[idxsel], Xset[idxsel]

def optim_func(X0, P1, P2, points_mat):

    lossval = 0

    u1, v1, u2, v2 = points_mat

    X0_h = np.append(X0, 1)

    lossval += ( u1 - (np.matmul(P1[0,:], X0_h)/ np.matmul(P1[2,:], X0_h)) )**2 + \
               ( v1 - (np.matmul(P1[1,:], X0_h)/ np.matmul(P1[2,:], X0_h)) )**2 + \
               ( u2 - (np.matmul(P2[0,:], X0_h)/ np.matmul(P2[2,:], X0_h)) )**2 + \
               ( v2 - (np.matmul(P2[1,:], X0_h)/ np.matmul(P2[2,:], X0_h)) )**2

    return lossval

def NonlinearTriangulation(X0, P0, P, points_RANSAC, max_nfev):

    X0_nloptim = np.zeros(X0.shape)
    residual = 0
    for i in range(X0.shape[0]):
        X0_flat = X0[i]
        res_lsq = least_squares(optim_func, X0_flat, args=(P0, P, points_RANSAC[i]), max_nfev=max_nfev)
        X0_nloptim[i,:] = res_lsq.x
        residual+=res_lsq.fun
    residual /= X0.shape[0]

    return X0_nloptim, residual

def Estimate3DPoints(imnum_1, imnum_2, txtpath):
    '''
    Estimate the fundamental matrix
    '''
    # Parse the provided text files to extract the corresponding points
    textfilename = txtpath+str(imnum_1)+'.txt'
    points_mat = np.zeros((1,4))
    with open(textfilename) as f:
        for idx,line in enumerate(f):
            if idx==0:
                n_features = int(line.strip().split(' ')[1])
            else:
                n_matches = int(line.strip().split(' ')[0])-1
                (ui,vi) = line.strip().split(' ')[4:6]
                ui = float(ui)
                vi = float(vi)
                corr_points = line.strip().split(' ')[6:]
                for i in range(n_matches):
                    corr_img = int(corr_points[3*i])
                    ui_d = float(corr_points[3*i+1])
                    vi_d = float(corr_points[3*i+2])
                    if corr_img == imnum_2:
                        points_mat = np.append(points_mat, [[ui, vi, ui_d, vi_d]], axis=0)

    points_mat = points_mat[1:,:]
    # points_mat contains the matching points between images imnum_1 and
    # imnum_2 in the format (u,v,u',v')

    # F = EstimateFundamentalMatrix(points_mat)

    '''
    Perform outlier rejection via RANSAC
    '''
    points_RANSAC = GetInlierRANSAC(points_mat)

    # Fundamental matrix using inliers
    F_RANSAC = EstimateFundamentalMatrix(points_RANSAC)

    '''
    Estimate essential matrix from the fundamental matrix
    '''
    # Set camera intrinsic parameters according to given file
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])

    E_recon = EssentialMatrixFromFundamentalMatrix(F_RANSAC, K)

    '''
    Estimate camera pose from essential matrix
    '''
    umat, smat, vmat = np.linalg.svd(E_recon)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    C1 = umat[:,2]
    R1 = np.matmul(np.matmul(umat, W), vmat)

    C2 = -1*umat[:,2]
    R2 = np.matmul(np.matmul(umat, W), vmat)

    C3 = umat[:,2]
    R3 = np.matmul(np.matmul(umat, W.T), vmat)

    C4 = -1*umat[:,2]
    R4 = np.matmul(np.matmul(umat, W.T), vmat)

    # Use C and R to calculate camera pose P
    # P = KR[I(3x3) -C]

    # P0 is the reference Camera
    P0 = ExtractCameraPose(K,np.zeros(C1.shape), np.identity(R1.shape[0]))
    P1 = ExtractCameraPose(K,C1,R1)
    P2 = ExtractCameraPose(K,C2,R2)
    P3 = ExtractCameraPose(K,C3,R3)
    P4 = ExtractCameraPose(K,C4,R4)

    # Identify the correct camera pose out of the 4 choices
    '''
    Linear triangulation to get 3D points
    '''
    # Get the reconstructed 3D points corresponding to each pose
    X1 = LinearTriangulation(P0, P1, points_RANSAC)
    X2 = LinearTriangulation(P0, P2, points_RANSAC)
    X3 = LinearTriangulation(P0, P3, points_RANSAC)
    X4 = LinearTriangulation(P0, P4, points_RANSAC)

    Cset = (C1, C2, C3, C4)
    Rset = (R1, R2, R3, R4)
    Xset = (X1, X2, X3, X4)

    # Check the cheirality condition to identify correct pose
    # C, R, X0 = DisambiguateCameraPose(Cset, Rset, Xset)
    C, R, X0 = DisambiguateCameraPose_2Check(Cset, Rset, Xset, K, points_RANSAC)
    P = ExtractCameraPose(K,C,R)

    '''
    Non-linear optimization of triangulation
    '''
    # X0_nloptim are the non-linearly optimized points
    X0_nloptim, residual = NonlinearTriangulation(X0, P0, P, points_RANSAC, max_nfev=100)

    return X0_nloptim, residual, F_RANSAC, E_recon, C, R, P
