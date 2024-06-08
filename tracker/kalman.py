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

class StateIndex(Enum):
    xl = 0
    vxl = 1
    yl = 2
    vyl = 3
    xc = 4
    vxc = 5
    yc = 6
    vyc = 7
    w = 8
    vw = 9
    h = 10
    vh = 11
    h1 = 12
    h4 = 13
    h7 = 14
    h2 = 15
    h5 = 16
    h8 = 17
    h3 = 18
    h6 = 19

class KalmanTracker(object):

    count = 1
    img_filter_indices = [StateIndex.xc.value, StateIndex.vxc.value, StateIndex.yc.value, StateIndex.vyc.value, StateIndex.w.value, StateIndex.vw.value, StateIndex.h.value, StateIndex.vh.value]
    ground_filter_indices = [StateIndex.xl.value, StateIndex.vxl.value, StateIndex.yl.value, StateIndex.vyl.value, StateIndex.h1.value, StateIndex.h4.value, StateIndex.h7.value, StateIndex.h2.value, StateIndex.h5.value, StateIndex.h8.value, StateIndex.h3.value, StateIndex.h6.value]
    # common_ground_indices = [StateIndex.xl.value, StateIndex.yl.value]
    # a_s = [-100, -10, -100, -10, -10e3, -150, -6e3, -150, 20, -50, 20, -50, -100, -100, -100, -100, -100, -100, -10e3, -6e3]
    # b_s = [100,   10,  100,  10,  10e3,  150,  6e3,  150, 300, 50, 500, 50,  100,  100,  100,  100,  100,  100,  10e3,  6e3]

    # aug_means = (np.asarray(a_s) + np.asarray(b_s)) / 2
    # aug_covs = np.square(np.asarray(b_s) - np.asarray(a_s)) / 12

    d_feet_d_img = np.array(  # xc, vxc, yc, vyc, w, vw, h, vh
            [[1, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 1, 0, 0, 0, 0.5, 0]]
        )
    
    # d_img_d_feet = np.array(
    #     [[1, 0, 0, 0, 0, 0,  0,  0,],
    #      [0, 0, 1, 0, 0, 0, -0.5,0]]
    # ).T

    def __init__(self, det, wx, wy, vmax,dt=1/30,H=None,H_P=None,H_Q=None,alpha_cov=1):
        # xl, vxl, yl, vyl, xc, vxc, yc, vyc, w, vw, h, vh, h1, h4, h7, h2, h5, h8, h3, h6
        self.kf = KalmanFilter(dim_x=20, dim_z=5)
        self.motion_transition_mat = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        self.A_orig = np.r_[H, [1]].reshape((3, 3)).T
        self.A = self.A_orig.copy()
        self.InvA_orig = np.linalg.inv(self.A_orig)

        det_feet, det_R = self.get_UV_and_error(det.get_box())
        x_local, r_local, _ = self.uv2xy(det_feet, det_R)

        self.kf.x[0] = x_local[0] #xl
        self.kf.x[1] = 0 #vxl
        self.kf.x[2] = x_local[1] #yl
        self.kf.x[3] = 0 #vyl
        self.kf.x[4] = det_feet[0] #xc
        self.kf.x[5] = 0 #vxc
        self.kf.x[6] = det_feet[1] - det.bb_height / 2 #yc
        self.kf.x[7] = 0 #vyc
        self.kf.x[8] = det.bb_width #w
        self.kf.x[9] = 0 #vw
        self.kf.x[10] = det.bb_height #h
        self.kf.x[11] = 0 #vh
        self.kf.x[-8:, 0] = H

        self._std_weight_position = 1 / 20 #* (dt / 0.04)
        self._std_weight_velocity = 1 / 160 * (dt / 0.04)

        std = [
                2 * self._std_weight_position * det.bb_width,
                10 * self._std_weight_velocity * det.bb_width,
                2 * self._std_weight_position * det.bb_height,
                10 * self._std_weight_velocity * det.bb_height,
                2 * self._std_weight_position * det.bb_width,
                10 * self._std_weight_velocity * det.bb_width,
                2 * self._std_weight_position * det.bb_height,
                10 * self._std_weight_velocity * det.bb_height
            ]

        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, (vmax / 3)**2, 1,  (vmax / 3)**2]))
        # np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))
        self.kf.P[[0, 2], 0] = (1 * r_local)[:, 0]
        self.kf.P[[0, 2], 2] = (1 * r_local)[:, 1]

        self.kf.P = scipy.linalg.block_diag(*(self.kf.P, np.diag(np.square(std)), H_P))
    
        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.Q = np.dot(np.dot(G, Q0), G.T)
        self.H_Q = H_Q

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1

        self.status = TrackStatus.Tentative
        self.alpha = alpha_cov

        self.dt = dt

        # Reference https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/IMM.py#L227
        mu = [0.5, 0.5]  # filter proba: ground, img
        self.mu = np.asarray(mu) / np.sum(mu)
        #M[i,j] is the probability of switching from filter j to filter i.
        self.M = np.array(
            [[0.7, 0.3],
             [0.3, 0.7]]
        )
        self.N = 2  # number of filters
        self.likelihood = np.zeros(self.N)
        # omega[i, j] is the probabilility of mixing the state of filter i into filter j
        self.omega = np.zeros((self.N, self.N))

        self._compute_mixing_probabilities()

    def _compute_mixing_probabilities(self):
        """
        Compute the mixing probability for each filter.
        """

        self.cbar = np.dot(self.mu, self.M)
        for i in range(self.N):
            for j in range(self.N):
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]

    def xy2uv(self, xy, sigma_xy):
        A = self.A

        xy = self.kf.x[[0, 2], :]
        xy1 = np.ones((3, 1))
        xy1[:2, :] = xy

        b = A @ xy1
        gamma = 1 / b[2,:]
        uv_proj = b[:2,:] * gamma

        dU_dA = gamma * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv_proj[0, 0], xy[1, 0], 0, -uv_proj[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv_proj[1, 0], 0, xy[1, 0], -uv_proj[1, 0] * xy[1, 0], 0, 1]
                                  ])
        dU_dX = np.zeros((2, 4))
        dU_dX[:, [0, 2]] = gamma * A[:2, :2] - (gamma**2) * b[:2,:] * A[2, :2]

        jac = np.c_[dU_dX, dU_dA]

        return uv_proj, jac @ sigma_xy @ jac.T, jac

    def uv2xy(self, uv, sigma_uv):
        uv1 = np.zeros((3, 1))
        uv1[:2,:] = uv
        uv1[2,:] = 1
        b = np.dot(self.InvA_orig, uv1)
        gamma = 1 / b[2,:]
        dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du
        xy = b[:2,:] * gamma

        R = np.dot(np.dot(dX_dU, sigma_uv[:2, :2]), dX_dU.T)
        return xy, R, dX_dU

    def update(self, y, R):
        A = self.A

        xy = self.kf.x[[0, 2], :]
        xy1 = np.ones((3, 1))
        xy1[:2, :] = xy

        b = A @ xy1
        gamma = 1 / b[2,:]
        uv_proj = b[:2,:] * gamma

        jacobian = np.zeros((5, self.kf.dim_x))

        dU_dA = gamma * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv_proj[0, 0], xy[1, 0], 0, -uv_proj[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv_proj[1, 0], 0, xy[1, 0], -uv_proj[1, 0] * xy[1, 0], 0, 1]
                                  ])
        
        dU_dX = gamma * A[:2, :2] - (gamma**2) * b[:2,:] * A[2, :2]

        dU_dU = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0.5]]
        )

        jacobian[:2, [0, 2]] = dU_dX
        jacobian[:2, -8:] = dU_dA
        jacobian[2:4, [StateIndex.xc.value, StateIndex.yc.value, StateIndex.w.value, StateIndex.h.value]] = dU_dU
        jacobian[-1, StateIndex.w.value] = 1

        diff = np.r_[y[2:4], y[2:]] - np.r_[uv_proj, uv_proj, self.kf.x[[StateIndex.w.value], :]]

        S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + R[2:4, 2:4]
        SI_0 = np.linalg.inv(S_0)

        S_1 = np.dot(jacobian[2:], np.dot(self.kf.P,jacobian[2:].T)) + R[2:, 2:]
        SI_1 = np.linalg.inv(S_1)

        # update mode probabilities from total probability * likelihood
        self.likelihood[0] = np.exp(
            -0.5 * (np.log(np.linalg.det(2*np.pi*S_0)) + diff[:2].T@SI_0@diff[:2])
        )
        self.likelihood[1] = np.exp(
            -0.5 * (np.log(np.linalg.det(2*np.pi*S_1)) + diff[2:].T@SI_1@diff[2:])
        )
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        self._compute_mixing_probabilities()

        kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0

        self.kf.x = self.kf.x + kalman_gain @ diff[:2]
        self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ jacobian[:2]) @ self.kf.P

        kalman_gain = self.kf.P @ jacobian[2:].T @ SI_1

        self.kf.x = self.kf.x + kalman_gain @ diff[2:]
        self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ jacobian[2:]) @ self.kf.P

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

    def compute_mixed_initial(self):
        # compute mixed initial conditions

        # Augment missing states of each filter with gaussian approximation of uniform dist.
        x_ground_aug = self.kf.x.copy()
        self.cov_ground_aug = self.kf.P.copy()

        x_img_aug = self.kf.x.copy()
        self.cov_img_aug = self.kf.P.copy()

        # Transform states so that they can be mixed
        ground_to_feet, P_d_feet_d_ground, jac_d_feet_d_ground = self.xy2uv(x_ground_aug[[StateIndex.xl.value, StateIndex.yl.value], :], 
                                                                    self.cov_ground_aug[self.ground_filter_indices, :][:, self.ground_filter_indices])
        P_d_img_d_ground = P_d_feet_d_ground # self.d_img_d_feet @ P_d_feet_d_ground @ self.d_img_d_feet.T  # 8x8
        x_ground_aug[StateIndex.xc.value, :] = ground_to_feet[0, :]
        self.cov_ground_aug[StateIndex.xc.value, StateIndex.xc.value] = P_d_img_d_ground[0, 0]

        feet_to_ground, P_d_ground_d_img, jac_d_ground_d_img = self.uv2xy(x_img_aug[[StateIndex.xc.value, StateIndex.yc.value], :] + np.asarray([[0], [x_img_aug[StateIndex.h.value, 0]]]) / 2,
                                                                          self.d_feet_d_img @ self.cov_img_aug[self.img_filter_indices, :][:, self.img_filter_indices] @ self.d_feet_d_img.T)
        x_img_aug[[StateIndex.xl.value, StateIndex.yl.value], :] = feet_to_ground
        self.cov_img_aug[[StateIndex.xl.value, StateIndex.yl.value], StateIndex.xl.value] = P_d_ground_d_img[:, 0]
        self.cov_img_aug[[StateIndex.xl.value, StateIndex.yl.value], StateIndex.yl.value] = P_d_ground_d_img[:, 1]

        # Mix
        omega = self.omega.T
        
        new_x_ground_aug = omega[0, 0] * x_ground_aug + omega[0, 1] * x_img_aug
        y1, y2 = x_ground_aug - new_x_ground_aug, x_img_aug - new_x_ground_aug
        new_cov_ground_aug = omega[0, 0] * (np.outer(y1, y1) + self.cov_ground_aug) + omega[0, 1] * (np.outer(y2, y2) + self.cov_img_aug)

        new_x_img_aug = omega[1, 0] * x_ground_aug + omega[1, 1] * x_img_aug
        y1, y2 = x_ground_aug - new_x_img_aug, x_img_aug - new_x_img_aug
        new_cov_img_aug = omega[1, 0] * (np.outer(y1, y1) + self.cov_ground_aug) + omega[1, 1] * (np.outer(y2, y2) + self.cov_img_aug)
        
        if np.isnan(new_x_ground_aug).any():
            print()

        self.kf.x[self.ground_filter_indices, :] = new_x_ground_aug[self.ground_filter_indices, :]
        self.kf.x[self.img_filter_indices, :] = new_x_img_aug[self.img_filter_indices, :]

        for idx in self.ground_filter_indices:
            self.kf.P[idx] = new_cov_ground_aug[idx]
        for idx in self.img_filter_indices:
            self.kf.P[idx] = new_cov_img_aug[idx]

    def predict(self, affine):
        # self.compute_mixed_initial()

        self.kf.F = scipy.linalg.block_diag(
            *(
                self.motion_transition_mat, 
                self.motion_transition_mat,
                self.motion_transition_mat,
                scipy.linalg.block_diag(
                    np.kron(np.eye(2), np.r_[affine, [[0, 0, 1]]]),
                    affine[:2, :2],
                )
            )
        )
        
        std = [
                self._std_weight_position * self.kf.x[StateIndex.w.value, 0],
                self._std_weight_velocity * self.kf.x[StateIndex.w.value, 0],
                self._std_weight_position * self.kf.x[StateIndex.h.value, 0],
                self._std_weight_velocity * self.kf.x[StateIndex.h.value, 0],
                self._std_weight_position * self.kf.x[StateIndex.w.value, 0],
                self._std_weight_velocity * self.kf.x[StateIndex.w.value, 0],
                self._std_weight_position * self.kf.x[StateIndex.h.value, 0],
                self._std_weight_velocity * self.kf.x[StateIndex.h.value, 0]
            ]

        motion_cov = np.diag(np.square(std))
        self.kf.Q = scipy.linalg.block_diag(
            self.Q, motion_cov, self.H_Q
        )
        
        self.kf.predict()
        self.kf.x[-2:, 0] += affine[:2, -1]

        R = affine[:2, :2]
        R8x8 = np.kron(np.eye(4, dtype=float), R)
        t = affine[:2, 2]

        indices = [StateIndex.xc.value, StateIndex.yc.value, StateIndex.w.value, StateIndex.h.value,
                   StateIndex.vxc.value, StateIndex.vyc.value, StateIndex.vw.value, StateIndex.vh.value]
        self.kf.x[indices, :] = R8x8 @ self.kf.x[indices, :]
        self.kf.x[[StateIndex.xc.value, StateIndex.yc.value], :] += t[:, None]

        new_cov = R8x8 @ self.kf.P[indices, :][:, indices] @ R8x8.T
        for num, idx in enumerate(indices):
            self.kf.P[idx, indices] = new_cov[num] 

        self.age += 1

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

    def get_state(self):
        return self.kf.x

    def get_UV_and_error(self, box):
        uv = np.array([[box[0]+box[2]/2], [box[1]+box[3]]])
        u_err,v_err = getUVError(box)
        sigma_uv = np.identity(2)
        sigma_uv[0,0] = u_err*u_err
        sigma_uv[1,1] = v_err*v_err
        return uv, sigma_uv
    
    def distance(self, y, R):
        b = np.dot(self.A, np.array([[self.kf.x[0, 0]], [self.kf.x[2, 0]], [1]]))
        uv = b / b[-1, 0]  # image proj
        gamma = 1 / b[2, :]  # gamma to image
        dU_dX = gamma * self.A[:2, :2] - (gamma**2) * b[:2,:] * self.A[2, :2]

        self_xy = np.array([[self.kf.x[0, 0]],
                            [self.kf.x[2, 0]]])
        dU_dA = gamma * np.array([
            [self_xy[0, 0], 0, -self_xy[0, 0] * uv[0, 0], self_xy[1, 0], 0, -uv[0, 0] * self_xy[1, 0], 1, 0],
            [0, self_xy[0, 0], -self_xy[0, 0] * uv[1, 0], 0, self_xy[1, 0], -uv[1, 0] * self_xy[1, 0], 0, 1]
                                  ])  # du/dA
        
        b = self.InvA_orig @ uv  # ground plane coords with H_L
        self_xy = b / b[-1, 0]
        gamma = 1 / b[2, :]
        dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du

        jacobian = np.zeros((5, self.kf.dim_x))
        jacobian[:2, [0, 2]] = dX_dU @ dU_dX
        jacobian[:2, -8:] = dX_dU @ dU_dA
        # jacobian[:2, [0, 2]] = dU_dX
        # jacobian[:2, -8:] = dU_dA

        jacobian[2:4, [StateIndex.xc.value, StateIndex.yc.value, StateIndex.w.value, StateIndex.h.value]] = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0.5]]
        )
        jacobian[-1,  StateIndex.w.value] = 1

        self_xy = np.r_[self_xy[:2, :], uv[:2, :], [self.kf.x[StateIndex.w.value]]]
        diff = (np.array(y) - self_xy)
        R = np.array(R)

        S1 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + R[:, :2, :2]
        SI = np.linalg.inv(S1)
        mahalanobis_1 = diff[:, :2].transpose(0, 2, 1) @ SI @ diff[:, :2]
        try:
            logdet1 = np.linalg.det(S1)
            logdet1 = np.log(logdet1)
        except (RuntimeWarning, LinAlgError):
            logdet1 = 6000
        logdet1[np.isnan(logdet1)] = 6000

        S2 = np.dot(jacobian[2:], np.dot(self.kf.P,jacobian[2:].T)) + R[:, 2:, 2:]
        SI = np.linalg.inv(S2)
        mahalanobis_2 = diff[:, 2:].transpose(0, 2, 1) @ SI @ diff[:, 2:]
        try:
            logdet2 = np.linalg.det(S2)
            logdet2 = np.log(logdet2)
        except (RuntimeWarning, LinAlgError):
            logdet2 = 6000
        logdet2[np.isnan(logdet2)] = 6000
        # return mahalanobis_1.squeeze() + mahalanobis_2.squeeze()
        return self.cbar[0] * (mahalanobis_1.squeeze() + logdet1) + self.cbar[1] * (mahalanobis_2.squeeze())
    