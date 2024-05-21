import warnings
from filterpy.kalman import KalmanFilter
from matplotlib.pylab import LinAlgError
import numpy as np
from enum import Enum
import scipy
import scipy.linalg 

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted   = 2

class KalmanTracker(object):

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w,h,dt=1/30,H=None,H_P=None,H_Q=None):
        
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


    # def update(self, y, R):
    #     self.kf.update(y,R)

    def predict_homog(self, affine):
        self.kf.F = scipy.linalg.block_diag(*(np.eye(4), 
                                              
                                              scipy.linalg.block_diag(
            np.kron(np.eye(2), np.r_[affine, [[0, 0, 1]]]),
            affine[:2, :2],
        )
                                              ))
        self.kf.predict()
        self.kf.x[-2:, 0] += affine[:2, -1]


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
        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        return self.kf.x
    
    def distance(self, y, R):
        # A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T

        # xy = self.kf.x[[0, 2], :]
        # xy1 = np.ones((3, 1))
        # xy1[:2, :] = xy

        # uv = y

        # b = A @ xy1
        # gamma = 1 / b[2,:]
        # uv_proj = b[:2,:] * gamma

        # dU_dA = gamma * np.array([
        #     [xy[0, 0], 0, -xy[0, 0] * uv_proj[0, 0], xy[1, 0], 0, -uv_proj[0, 0] * xy[1, 0], 1, 0],
        #     [0, xy[0, 0], -xy[0, 0] * uv_proj[1, 0], 0, xy[1, 0], -uv_proj[1, 0] * xy[1, 0], 0, 1]
        #                           ])
        
        # dU_dX = np.zeros((2, self.kf.dim_x - 8))
        # dU_dX[:, [0, 2]] = gamma * A[:2, :2] - (gamma**2) * b[:2,:] * A[2, :2]

        # diff = uv - uv_proj

        # jacobian = np.c_[dU_dX, dU_dA]

        # S = np.dot(jacobian, np.dot(self.kf.P,jacobian.T)) + R
        # SI = np.linalg.inv(S)
        # mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        # try:
        #     logdet = np.linalg.det(S)
        #     logdet = np.log(logdet)
        # except (RuntimeWarning, LinAlgError):
        #     logdet = 1000
        # return mahalanobis[0,0] #+ logdet

        diff = y - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        try:
            logdet = np.log(np.linalg.det(S))
        except (LinAlgError, RuntimeWarning):
            logdet = 1000
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
