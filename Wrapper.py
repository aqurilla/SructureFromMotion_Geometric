#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 3: Buildings built in minutes - An SfM Approach

References:

"""

# Code starts here:

# Imports
import numpy as np
import cv2
import argparse
import pdb
import glob
from misc.utils import *
from misc.helpers import *
from scipy.optimize import least_squares


# Camera intrinsic params
K = np.array([[568.996140852, 0, 643.21055941],
			  [0, 568.988362396, 477.982801038],
			  [0, 0, 1]])

def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	#Parser.add_argument('--imnum_1', default=1, help='Image number of 1st image')
	#Parser.add_argument('--imnum_2', default=2, help='Image number of 2nd image')
	Parser.add_argument('--txtpath', default='../Data/matching', help='Prefix path to correspondence files')
	Parser.add_argument('--imgpath', default='../Data/', help='Prefix path to image files')

	Args = Parser.parse_args()
	#imnum_1 = int(Args.imnum_1)
	#imnum_2 = int(Args.imnum_2)
	txtpath = Args.txtpath
	imgpath = Args.imgpath

	# X0_nloptim, residual, F_RANSAC, E_recon, C, R, P = Estimate3DPoints(imnum_1, imnum_2, txtpath)

	'''
	Perform outlier rejection via RANSAC and estimate the fundamental matrix
	'''
	# Parse correspondence points between the first two images
	points_mat = parsePoints(1, 2, txtpath)

	points_RANSAC = GetInlierRANSAC(points_mat)

	# Display correspondence image
	# drawCorrespondences(1,2,points_RANSAC, imgpath)

	# Fundamental matrix using inliers
	F_RANSAC = EstimateFundamentalMatrix(points_RANSAC)

	'''
	Estimate essential matrix from the fundamental matrix
	'''
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
	# X0_nl are the non-linearly optimized points
	X0_nl, residual = NonlinearTriangulation(X0, P0, P, points_RANSAC, max_nfev=100)

	# Display triangulation points obtained
	# dispTriangulation(X0, X0_nl, P)

	Cset = []
	Rset = []
	Cset.append(C)
	Rset.append(R)

	CRCs = np.hstack([R,C.reshape([3,1])])

	im2world = impts2wpts(dict(),1,X0_nl,points_RANSAC[:,:2])
	im2world = impts2wpts(im2world,2,X0_nl,points_RANSAC[:,2:])

	for i in range(2,6):
		print("Image #"+str(i))
		points_mat = parsePoints(i, i+1, txtpath) # Parse points from matchingi.txt
		points_RANSAC = GetInlierRANSAC(points_mat)

		X,x = getWorldPts(im2world,i,points_RANSAC)

		X_RANSAC, x_RANSAC , RC = PnPRANSAC(X, x, K)
		print(X_RANSAC.shape)
		print(x_RANSAC.shape)
		RC_nl = NonlinearPnP(X_RANSAC, x_RANSAC, K, RC)
		Rnew = RC_nl[:,:3]
		Cnew = RC_nl[:,-1]
		Cset.append(Cnew)
		Rset.append(Rnew)

		CRCs = CRCs.stack([CRCs,RC_nl.reshape([3,4,1])],2)
		
		Pnew = ExtractCameraPose(K,Cnew,Rnew)
		Xnew = LinearTriangulation(P0, P, points_RANSAC)
		Xnew_nl, residual = NonlinearTriangulation(Xnew, P0, Pnew, points_RANSAC, max_nfev=100)
		
		pdb.set_trace()
		
		im2world = impts2wpts(im2world, i+1, Xnew_nl, x_RANSAC)
		# X = np.append(X, Xnew_nl, axis=0)

		# BuildVisibilityMatrix
		V,X,x = BuildVisibilityMatrix(6,im2world)

		# BundleAdjustment
		newcams, newpts, info = BundleAdjustment(X,x,K,CRCs,V)

		pdb.set_trace()

if __name__ == '__main__':
	main()
