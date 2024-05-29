import warnings
from filterpy.kalman import KalmanFilter
from matplotlib.pylab import LinAlgError
import numpy as np
from enum import Enum
import scipy
import scipy.linalg

from detector.mapper import getUVError 

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted   = 2

class KalmanTracker(object):

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w,h,dt=1/30,H=None,H_P=None,H_Q=None,alpha_cov=1):
        
        self.kf = KalmanFilter(dim_x=12, dim_z=2)
        self.motion_transition_mat = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.kf.H = np.c_[np.array([[1, 0, 0, 0], [0, 0, 1, 0]]), np.zeros((2, 8))]
        self.kf.R = R
        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, (vmax / 3)**2, 1,  (vmax / 3)**2]))
        # np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))
        self.kf.P[[0, 2], 0] = (1 * R)[:, 0]
        self.kf.P[[0, 2], 2] = (1 * R)[:, 1]
        self.kf.P = scipy.linalg.block_diag(*(self.kf.P, H_P))
    
        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.kf.Q = np.dot(np.dot(G, Q0), G.T)
        self.kf.Q = scipy.linalg.block_diag(*(self.kf.Q, H_Q))

        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = y[1]
        self.kf.x[3] = 0
        self.kf.x[4:, 0] = H

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.w = w
        self.h = h

        self.status = TrackStatus.Tentative
        self.alpha = alpha_cov


    def update(self, y, R):
        A = self.A

        xy = self.kf.x[[0, 2], :]
        xy1 = np.ones((3, 1))
        xy1[:2, :] = xy

        uv = y

        b = A @ xy1
        gamma = 1 / b[2,:]
        uv_proj = b[:2,:] * gamma

        dU_dA = gamma * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv_proj[0, 0], xy[1, 0], 0, -uv_proj[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv_proj[1, 0], 0, xy[1, 0], -uv_proj[1, 0] * xy[1, 0], 0, 1]
                                  ])
        
        dU_dX = np.zeros((2, self.kf.dim_x - 8))
        dU_dX[:, [0, 2]] = gamma * A[:2, :2] - (gamma**2) * b[:2,:] * A[2, :2]

        diff = uv - uv_proj

        jacobian = np.c_[dU_dX, dU_dA]

        S = np.dot(jacobian, np.dot(self.kf.P,jacobian.T)) + R
        SI = np.linalg.inv(S)

        kalman_gain = self.kf.P @ jacobian.T @ SI

        # print(np.isclose(self.kf.x[:4], (self.kf.x + kalman_gain @ (homog[:, None] - proj))[:4]).all())
        self.kf.x = self.kf.x + kalman_gain @ diff
        self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ jacobian) @ self.kf.P

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        self.kf.Q[-8:, -8:] = self.alpha * self.kf.Q[-8:, -8:] + ((1 - self.alpha) * kalman_gain @ diff @ diff.T @ kalman_gain.T)[-8:, -8:]

    def predict(self, affine):
        self.kf.F = scipy.linalg.block_diag(*(self.motion_transition_mat, 
                                              scipy.linalg.block_diag(
            np.kron(np.eye(2), np.r_[affine, [[0, 0, 1]]]),
            affine[:2, :2],
        )
                                              ))
        self.kf.predict()
        self.kf.x[-2:, 0] += affine[:2, -1]

        self.age += 1

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        return self.kf.x
    
    def uv2xy(self, uv, sigma_uv):
        uv1 = np.zeros((3, 1))
        uv1[:2,:] = uv
        uv1[2,:] = 1
        b = np.dot(self.InvA, uv1)
        gamma = 1 / b[2,:]
        dX_dU = gamma * self.InvA[:2, :2] - (gamma**2) * b[:2,:] * self.InvA[2, :2]  # dX/du
        xy = b[:2,:] * gamma

        dU_dA = gamma * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv[0, 0], xy[1, 0], 0, -uv[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv[1, 0], 0, xy[1, 0], -uv[1, 0] * xy[1, 0], 0, 1]
                                  ])  # du/dA
        dX_dA = dX_dU @ dU_dA

        sigma_xy = np.dot(np.dot(dX_dU, sigma_uv), dX_dU.T) + np.dot(np.dot(dX_dA, self.kf.P[-8:, -8:]), dX_dA.T)
        return xy, sigma_xy
    
    def mapto(self,box):
        uv, sigma_uv = self.get_UV_and_error(box)
        y,R = self.uv2xy(uv, sigma_uv)
        return y,R

    def get_UV_and_error(self, box):
        uv = np.array([[box[0]+box[2]/2], [box[1]+box[3]]])
        u_err,v_err = getUVError(box)
        sigma_uv = np.identity(2)
        sigma_uv[0,0] = u_err*u_err
        sigma_uv[1,1] = v_err*v_err
        return uv, sigma_uv
    
    def distance(self, y, R):
        A = self.A

        xy = self.kf.x[[0, 2], :]
        xy1 = np.ones((3, 1))
        xy1[:2, :] = xy

        uv = y

        b = A @ xy1
        gamma = 1 / b[2,:]
        uv_proj = b[:2,:] * gamma

        dU_dA = gamma * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv_proj[0, 0], xy[1, 0], 0, -uv_proj[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv_proj[1, 0], 0, xy[1, 0], -uv_proj[1, 0] * xy[1, 0], 0, 1]
                                  ])
        
        dU_dX = np.zeros((2, self.kf.dim_x - 8))
        dU_dX[:, [0, 2]] = gamma * A[:2, :2] - (gamma**2) * b[:2,:] * A[2, :2]

        diff = uv - uv_proj

        jacobian = np.c_[dU_dX, dU_dA]

        S = np.dot(jacobian, np.dot(self.kf.P,jacobian.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        try:
            logdet = np.linalg.det(S)
            logdet = np.log(logdet)
        except (RuntimeWarning, LinAlgError):
            logdet = 6000
        if np.isnan(logdet):
            logdet = 6000
        return mahalanobis[0,0] + logdet / 6

        xy, R = self.uv2xy(y, R)
        jacobian = np.zeros((2, self.kf.dim_x))
        jacobian[:, [0, 2]] = np.eye(2)
        
        diff = xy - self.kf.x[[0, 2], :]

        S = np.dot(jacobian, np.dot(self.kf.P,jacobian.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        try:
            logdet = np.linalg.det(S)
            logdet = np.log(logdet)
        except (RuntimeWarning, LinAlgError):
            logdet = 6000
        if np.isnan(logdet):
            logdet = 6000
        return mahalanobis[0,0] + logdet
    
    def update_homography(self, homog, homog_cov):
        update_mat = np.zeros((homog.size, self.kf.dim_x))
        homog_update_mat = np.eye(8)
        update_mat[:, -8:] = homog_update_mat

        proj = update_mat @ self.kf.x
        proj_cov = update_mat @ self.kf.P @ update_mat.T + homog_cov

        kalman_gain = self.kf.P @ update_mat.T @ np.linalg.inv(proj_cov)

        # print(np.isclose(self.kf.x[:4], (self.kf.x + kalman_gain @ (homog[:, None] - proj))[:4]).all())
        self.kf.x = self.kf.x + kalman_gain @ (homog[:, None] - proj)
        self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ update_mat) @ self.kf.P
