import numpy as np
from random import sample,choice
from scipy.optimize import leastsq
from cv2 import Rodrigues
from scipy.spatial.transform import Rotation as Rot
import sba
import pdb


def LinearPnP(X,x,K):
	# X is 4x6, x is 3x6, K ix 3x3
	RC = np.linalg.lstsq(X.transpose(),np.matmul(np.linalg.inv(K),x).transpose(),rcond=None)[0].transpose()
	R = RC[:,:-1]
	if np.linalg.det(R) < 0:
		R = -R
		RC[:,:-1] = R
	RC[:,-1] = np.matmul(-R.transpose(),RC[:,-1])
	return RC

def impts2wpts(curDict,camidx,newX,newx):
	for idx,pt in enumerate(newx):
		curDict[(camidx,pt[0],pt[1])] = newX[idx,:]
	return curDict

def getWorldPts(im2world,fromIdx,ptCorresps):
	x = np.zeros((2,1))
	X = np.zeros((3,1))
	for ptc in ptCorresps:
		tup = (fromIdx,ptc[0],ptc[1])
		if tup in im2world:
			x= np.hstack((x,ptc[2:].reshape((2,1))))
			X= np.hstack((X,im2world[tup].reshape((3,1))))
	x = x[:,1:]
	X = X[:,1:]
	return X,x

def reprojErr(X,x,P):
	# X is 4xN, x is 2xN, P is 3x3
	PX = np.matmul(P,X)
	fracs = np.vstack([PX[0,:]/PX[2,:],PX[1,:]/PX[2,:]])
	return sum((x - fracs)**2,0)

def PnPRANSAC(X,x,K,eps=500000,numIters=2000):
	# X is 3xN, x is 2xN, K is 3x3
	l = X.shape[1]
	sampleids = range(l)
	Xhat = np.vstack([X,np.ones([1,l])])
	xhat = np.vstack([x,np.ones([1,l])])
	bestInliers = np.zeros(l,dtype=np.bool)
	bestRC = np.zeros([3,4])
	for _ in range(numIters):
		pointids = sample(sampleids,6)
		Xs = Xhat[:,pointids]
		xs = xhat[:,pointids]
		RC = LinearPnP(Xs,xs,K)
		P = np.matmul(K,RC*np.array([1,1,1,-1]))
		re = reprojErr(Xhat,x,P)
		inliers = np.less(re,eps)
		if sum(inliers) > sum(bestInliers):
			bestInliers = inliers
			bestRC = RC
	return bestRC

def NonlinearPnP(X,x,K,RC):
	# X is 3xN, x is 2xN, K is 3x3, RC is 3x4
	Xhat = np.vstack([X,np.ones([1,X.shape[1]])])
	C = RC[:,-1]
	R = RC[:,:-1]
	def obj(qc,X,x):
		R = q2R(qc[:4])
		C = qc[4:].reshape([3,1])
		return reprojErr(X,x,np.matmul(K,np.concatenate([R,-C],1)))
	qc = leastsq(obj,np.concatenate([R2q(R),C]),args=(Xhat,x))[0]
	return np.concatenate([q2R(qc[:4]),qc[4:].reshape([3,1])],1)

def BuildVisibilityMatrix(nCams,im2world):
	wcs = np.unique(np.array(list(im2world.values())),axis=0)
	wc2idx = dict(zip([tuple(wc) for wc in wcs],np.arange(wcs.size,dtype=np.int32)))
	V = np.zeros([wcs.shape[0],nCams])
	x = np.zeros([wcs.shape[0],nCams,2])
	for ((camidx,u,v),wc) in im2world.items():
		row = wc2idx[tuple(wc)]
		V[row,camidx-1] = 1
		x[row,camidx-1,0] = u
		x[row,camidx-1,1] = v
	return V,wcs,x

def R2r(R):
	out,_ = Rodrigues(R,np.zeros(3))
	return out

def r2R(r):
	out,_ = Rodrigues(r,np.zeros([3,3]))
	return out

def r2q(r):
	robj = Rot.from_rotvec(r)
	return robj.as_quat()

def q2R(q):
	robj = Rot.from_quat(q)
	return r2R(robj.as_rotvec())

def R2q(R):
	return r2q(R2r(R))

def Cam2sba(n, CRCs, K):
	# n - number of cameras
	sbainp = np.zeros((n, 17))

	sbainp[:,0] = K[0][0] # fx
	sbainp[:,1] = K[0][2] # cx
	sbainp[:,2] = K[1][2] # cy
	sbainp[:,3] = 1       # AR
	sbainp[:,4] = K[0][1] # s

	for i in range(n):
		R = (CRCs[:,:,i])[:,:3]
		C = (CRCs[:,:,i])[:,3]
		sbainp[i,10:14] = R2q(R)
		sbainp[i,14:] = C.reshape((1,3))

	return sbainp

def sba2Cam(newcams):
	pdb.set_trace()
	CRCs = np.zeros((3,4,newcams.shape[0]))
	for i in range(newcams.shape[0]):
		R = q2R(newcams[i,10:14])
		C = newcams[i,14:]
		CRCs[:,:3,i] = R
		CRCs[:,3,i] = C

	return CRCs

def BundleAdjustment(X,x,K,CRCs,V):
	# X is 3xN, x is 2xN, CKs is 3x3xI, CRCs is 3x4xI, V is IxN
	pts = sba.Points(X,x,V)
	cams = sba.Cameras.fromDylan(Cam2sba(CRCs.shape[2], CRCs, K))
	newcams, newpts, info = sba.SparseBundleAdjust(cams,pts)
	newcams = sba2Cam(toDyl(newcams))
	newpts = newpts._getB()
	return newcams, newpts, info

def toDyl(cams):
	result = np.zeros((cams.ncameras, 17),dtype=np.double)
	result[:,0:10] = cams.camarray[:,0:10]
	result[:,-6:] = cams.camarray[:,-6:]

	# construct missing q0 real part of unit quaternions
	for cam in range(cams.ncameras):
		result[cam,-7] = np.sqrt(1-cams.camarray[cam,-6]**2.
                                 -cams.camarray[cam,-5]**2.
                                 -cams.camarray[cam,-4]**2.)

	return result
