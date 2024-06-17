import warnings
from filterpy.kalman import KalmanFilter
from matplotlib.pylab import LinAlgError
import numpy as np
from enum import Enum
import scipy
import scipy.linalg

from detector.mapper import Mapper, getUVError 

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted   = 2

class StateIndex(Enum):
    xl = 0
    vxl = 1
    axl = 2
    yl = 3
    vyl = 4
    ayl = 5
    h1 = 6
    h4 = 7
    h7 = 8
    h2 = 9
    h5 = 10
    h8 = 11
    h3 = 12
    h6 = 13

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  

class KalmanTracker(object):

    count = 1

    def __init__(self, det, wx, wy, vmax,dt=1/30,H=None,H_P=None,H_Q=None,alpha_cov=1,InvA_orig=None,maneuver_time=2):
        # xl, vxl, axl, yl, vyl, ayl, h1, h4, h7, h2, h5, h8, h3, h6
        self.alpha = alpha = 1 / maneuver_time

        self.kf = KalmanFilter(dim_x=14, dim_z=2)
        self.motion_transition_mat = np.array([[1, dt, 1 / (alpha**2) * (-1 + alpha * dt + np.e**(-alpha*dt))], 
                                               [0, 1, 1 / alpha * (1 - np.e**(-alpha*dt))], 
                                               [0, 0, np.e**(-alpha*dt)]])
        self.motion_transition_mat = scipy.linalg.block_diag(self.motion_transition_mat, self.motion_transition_mat)

        self.A = np.r_[H, [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)
        self.InvA_orig = InvA_orig

        det_feet, det_R = self.get_UV_and_error(det.get_box())
        x_local, r_local, _ = self.uv2xy(det_feet, det_R, H_P)

        self.R = scipy.linalg.block_diag(r_local, det_R)

        self.kf.x[StateIndex.xl.value] = x_local[0] #xl
        self.kf.x[StateIndex.vxl.value] = 0 #vxl
        self.kf.x[StateIndex.axl.value] = 0 #vxl
        self.kf.x[StateIndex.yl.value] = x_local[1] #yl
        self.kf.x[StateIndex.vyl.value] = 0 #vyl
        self.kf.x[StateIndex.ayl.value] = 0 #vyl
        self.kf.x[-8:, 0] = H

        self.kf.P = np.zeros((6, 6))
        np.fill_diagonal(self.kf.P, np.array([1, (vmax / 3)**2, (wx / 3) ** 2, 1,  (vmax / 3)**2, (wy / 3) ** 2]))
        self.kf.P[[StateIndex.xl.value, StateIndex.yl.value], StateIndex.xl.value] = (1 * r_local)[:, 0]
        self.kf.P[[StateIndex.xl.value, StateIndex.yl.value], StateIndex.yl.value] = (1 * r_local)[:, 1]

        self.kf.P = scipy.linalg.block_diag(*(self.kf.P, H_P))

        self.Q =np.array([
            [1 / (2 * alpha**5) * (1 - np.e**(-2*alpha*dt) + 2*alpha*dt + 2*alpha**3*dt**3/3 - 2*alpha**2*dt**2 - 4*alpha*dt*np.e**(-alpha*dt)), 1/(2*alpha**4) * (np.e**(-2*alpha*dt) + 1 - 2*np.e**(-alpha*dt) + 2*alpha*dt*np.e**(-alpha*dt) - 2*alpha*dt + alpha**2*dt**2), 1/(2*alpha**3)*(1 - np.e**(-2*alpha*dt) - 2*alpha*dt*np.e**(-alpha*dt))],
            [0, 1/(2*alpha**3)*(4*np.e**(-alpha*dt)-3-np.e**(-2*alpha*dt)+2*alpha*dt), 1/(2**alpha**2)*(np.e**(-2*alpha*dt) + 1 -2*np.e**(-alpha*dt))],
            [0, 0, 1/(2*alpha) * (1 - np.e**(-2*alpha*dt))]
        ])
        self.Q[1,0] = self.Q[0,1]
        self.Q[2,0] = self.Q[0,2]
        self.Q[2,1] = self.Q[1,2]
        self.Q = scipy.linalg.block_diag(2 * alpha * wx ** 2 * self.Q, 2 * alpha * wy ** 2 * self.Q)
        self.H_Q = H_Q

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1

        self.status = TrackStatus.Tentative
        self.alpha = alpha_cov

        self.innov_list = []
        self.residual_list = []
        box_array = np.array([det.bb_left, det.bb_top, det.bb_left + det.bb_width, det.bb_top + det.bb_height])
        self.box_buffer = [box_array]
        self.box_pred = box_array
        self.relative_iou = 1
        self.g_mahala = 0

        self.dt = dt
        self.uv =  np.array([det.foot_x, det.foot_y]).astype(int)

        # Reference https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/IMM.py#L227
        mu = [0.5, 0.5]  # filter proba: ground, img
        self.mu = np.asarray(mu) / np.sum(mu)
        #M[i,j] is the probability of switching from filter j to filter i.
        self.M = np.array(
            [[0.9, 0.1],
             [0.1, 0.9]]
        )
        self.N = 2  # number of filters
        self.likelihood = np.zeros(self.N)
        # omega[i, j] is the probabilility of mixing the state of filter i into filter j
        self.omega = np.zeros((self.N, self.N))

        # self._compute_mixing_probabilities()

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

    def uv2xy(self, uv, sigma_uv, H_P):
        uv1 = np.zeros((3, 1))
        uv1[:2,:] = uv
        uv1[2,:] = 1
        b = np.dot(self.InvA, uv1)
        gamma = 1 / b[2,:]
        dX_dU = gamma * self.InvA[:2, :2] - (gamma**2) * b[:2,:] * self.InvA[2, :2]  # dX/du
        xy = b[:2,:] * gamma

        gamma2 = 1 / (np.dot(self.A[-1, :2], xy[:, 0]) + 1)
        dU_dA = gamma2 * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv[0, 0], xy[1, 0], 0, -uv[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv[1, 0], 0, xy[1, 0], -uv[1, 0] * xy[1, 0], 0, 1]
                                  ])  # du/dA
        dX_dA = dX_dU @ dU_dA

        R = np.dot(np.dot(dX_dU, sigma_uv[:2, :2]), dX_dU.T) + np.dot(np.dot(dX_dA, H_P), dX_dA.T)

        return xy, R, dX_dU

    def update(self, y, R, relative_iou, relative_p):
        A = self.A

        xy = self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :]
        xy1 = np.ones((3, 1))
        xy1[:2, :] = xy

        b = A @ xy1
        gamma = 1 / b[2,:]
        uv_proj = b[:2,:] * gamma

        jacobian = np.zeros((2, self.kf.dim_x))

        dU_dA = gamma * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv_proj[0, 0], xy[1, 0], 0, -uv_proj[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv_proj[1, 0], 0, xy[1, 0], -uv_proj[1, 0] * xy[1, 0], 0, 1]
                                  ])
        
        dU_dX = gamma * A[:2, :2] - (gamma**2) * b[:2,:] * A[2, :2]

        jacobian[:2, [StateIndex.xl.value, StateIndex.yl.value]] = dU_dX
        jacobian[:2, -8:] = dU_dA
        diff = y[2:4] - uv_proj

        S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + self.R[2:4, 2:4]
        SI_0 = np.linalg.inv(S_0)

        mahala1 = diff[:2].T@SI_0@diff[:2]
        self.likelihood[0] = relative_p
        # np.exp(
        #     -0.5 * (np.log(np.linalg.det(2*np.pi*S_0)) + mahala1)
        # )
        self.g_mahala = relative_p
        self.likelihood[1] = relative_iou
        self.relative_iou = relative_iou
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        if mahala1 < np.inf:
            kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0

            self.kf.x = self.kf.x + kalman_gain @ diff[:2]
            self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ jacobian[:2]) @ self.kf.P

            if self.death_count > 1 or not len(self.innov_list):
                self.innov_list = [diff[:2] @ diff[:2].T]
            else:
                self.innov_list.append(diff[:2] @ diff[:2].T)
                if len(self.innov_list) > 5:
                    self.innov_list.pop(0)
            new_pcov = (kalman_gain @ np.mean(self.innov_list, axis=0) @ kalman_gain.T)
            self.H_Q = new_pcov[-8:, -8:]

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        xy1[:2, :] = self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :]

        b = self.A @ xy1
        gamma = 1 / b[2,:]
        uv_proj = b[:2,:] * gamma

        self.uv = uv_proj[:2, 0].astype(int)

        residual = y[2:4] - uv_proj

        if self.death_count > 1 or not len(self.residual_list):
            self.residual_list = [residual @ residual.T]
        else:
            self.residual_list.append(residual @ residual.T)
            if len(self.residual_list) > 5:
                self.residual_list.pop(0)
        if self.death_count > 1:
            self.box_buffer = [y[4:8].squeeze()]
        else:
            self.box_buffer.append(y[4:8].squeeze())
            if len(self.box_buffer) > 5:
                self.box_buffer.pop(0)

        self.R = np.mean(self.residual_list, axis=0) + (S_0 - self.R[2:4, 2:4])
        b = np.dot(self.InvA_orig, np.r_[uv_proj, [[1]]])
        gamma = 1 / b[2,:]
        dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du
        self.R = scipy.linalg.block_diag(dX_dU @ self.R[:2, :2] @ dX_dU.T, self.R)

    def compute_mixed_initial(self):
        # compute mixed initial conditions

        # Transform states so that they can be mixed
        last_feet = self.box_buffer[-1].copy()
        last_feet[0] += (last_feet[2] - last_feet[0]) / 2
        last_feet[1] += (last_feet[3] - last_feet[1])
        last_feet = last_feet[:2, None]

        g_pos, sigma_rg, jac = self.uv2xy(last_feet, np.array([[13 ** 2, 0], [0, 10 ** 2]]), self.kf.P[-8:, -8:])
        

        # Mix
        omega = self.omega.T
        
        new_x = omega[0, 0] * self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :] + omega[0, 1] * g_pos
        y1, y2 = self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :] - new_x, g_pos - new_x
        new_cov = omega[0, 0] * (np.outer(y1, y1) + self.kf.P[[StateIndex.xl.value, StateIndex.yl.value], :][:, [StateIndex.xl.value, StateIndex.yl.value]]) + omega[0, 1] * (np.outer(y2, y2) + sigma_rg)

        self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :] = new_x
        self.kf.P[[StateIndex.xl.value, StateIndex.yl.value], StateIndex.xl.value] = new_cov[:, 0]
        self.kf.P[[StateIndex.xl.value, StateIndex.yl.value], StateIndex.yl.value] = new_cov[:, 1]


    def predict(self, affine):
        # if self.death_count == 1:
        self._compute_mixing_probabilities()
        self.compute_mixed_initial()

        self.kf.F = scipy.linalg.block_diag(
            *(
                self.motion_transition_mat, 
                scipy.linalg.block_diag(
                    np.kron(np.eye(2), np.r_[affine, [[0, 0, 1]]]),
                    affine[:2, :2],
                )
            )
        )

        self.kf.Q = scipy.linalg.block_diag(
            self.Q, self.H_Q
        )
        
        self.kf.predict()
        self.kf.x[-2:, 0] += affine[:2, -1]

        self.age += 1

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        b = np.dot(self.A, np.array([[self.kf.x[StateIndex.xl.value, 0]], [self.kf.x[StateIndex.yl.value, 0]], [1]]))
        self.uv = (b[:2, 0] / b[-1, 0]).astype(int)  # image proj

        if len(self.box_buffer) > 1:
            self.box_pred = self.box_buffer[-1] + self.death_count * np.mean(
                np.array(self.box_buffer)[1:] - np.array(self.box_buffer[:-1]), 
                axis=0)

    def get_state(self):
        return self.kf.x

    def get_UV_and_error(self, box):
        uv = np.array([[box[0]+box[2]/2], [box[1]+box[3]]])
        u_err,v_err = getUVError(box, Mapper.sigma_m)
        sigma_uv = np.identity(2)
        sigma_uv[0,0] = u_err*u_err
        sigma_uv[1,1] = v_err*v_err
        return uv, sigma_uv
    
    def distance(self, y, R, buf=0.3):
        b = np.dot(self.A, np.array([[self.kf.x[StateIndex.xl.value, 0]], [self.kf.x[StateIndex.yl.value, 0]], [1]]))
        uv = b / b[-1, 0]  # image proj
        gamma = 1 / b[2, :]  # gamma to image
        dU_dX = gamma * self.A[:2, :2] - (gamma**2) * b[:2,:] * self.A[2, :2]

        self_xy = np.array([[self.kf.x[StateIndex.xl.value, 0]],
                            [self.kf.x[StateIndex.yl.value, 0]]])
        dU_dA = gamma * np.array([
            [self_xy[0, 0], 0, -self_xy[0, 0] * uv[0, 0], self_xy[1, 0], 0, -uv[0, 0] * self_xy[1, 0], 1, 0],
            [0, self_xy[0, 0], -self_xy[0, 0] * uv[1, 0], 0, self_xy[1, 0], -uv[1, 0] * self_xy[1, 0], 0, 1]
                                  ])  # du/dA
        
        b = self.InvA_orig @ uv  # ground plane coords with H_L
        self_xy = b / b[-1, 0]
        gamma = 1 / b[2, :]
        dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du

        jacobian = np.zeros((2, self.kf.dim_x))
        jacobian[:2, [StateIndex.xl.value, StateIndex.yl.value]] = dX_dU @ dU_dX
        jacobian[:2, -8:] = dX_dU @ dU_dA

        y = np.array(y)
        diff = (y[:, :2] - self_xy[:2])
        # R = np.array(R)

        S1 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + np.repeat(np.expand_dims(self.R[:2, :2], 0), len(y), 0)
        SI = np.linalg.inv(S1)
        mahalanobis_1 = diff[:, :2].transpose(0, 2, 1) @ SI @ diff[:, :2]
        try:
            logdet1 = np.linalg.det(S1)
            logdet1 = np.log(logdet1)
        except (RuntimeWarning, LinAlgError):
            logdet1 = 6000
        logdet1[np.isnan(logdet1)] = 6000

        # ious = iou_batch(self.box_pred[None, :], np.atleast_2d(y[:, 4:8].squeeze()))
        # weighted_iou = ious * ious / (ious.sum() if ious.sum() > 0 else 1)

        buffered_y = np.c_[y[:, 4] - buf * y[:, 6], y[:, 5] - buf * y[:, 7], y[:, 6] + 2*buf * y[:, 6], y[:, 7] + 2*buf * y[:, 7]]
        bious = iou_batch(np.array([
            self.box_pred[0] - buf * self.box_pred[2],
            self.box_pred[1] - buf * self.box_pred[3],
            self.box_pred[2] + 2*buf*self.box_pred[2],
            self.box_pred[3] + 2*buf*self.box_pred[3],
        ])[None, :], buffered_y)
        # weighted_biou = bious * bious / (bious.sum() if bious.sum() > 0 else 1)
        # mahalanobis_2 = np.clip(scipy.stats.chi2.ppf(1 - weighted_biou, df=2), 0, 1000)#(1 - ious) * 7

        proba_1 = 1 - scipy.stats.chi2.cdf(mahalanobis_1.squeeze() + logdet1, df=2)
        # proba_1 = proba_1 * proba_1 / (proba_1.sum() if proba_1.sum() > 0 else 1)

        proba = (self.cbar[0] * proba_1 * bious + self.cbar[1] * bious) #* np.e ** (-(self.death_count - 1)/5)
        # proba = proba_1 * bious

        return 1 - proba, bious, proba_1
    