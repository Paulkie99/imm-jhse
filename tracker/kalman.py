import queue
import sys
from filterpy.kalman import KalmanFilter, IMMEstimator
from matplotlib.pylab import LinAlgError
import numpy as np
from enum import Enum
import scipy
from util.stats import logpdf
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

class StateIndexCV(Enum):
    xl = 0
    vxl = 1
    yl = 2
    vyl = 3

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

class SingerKalmanTracker(object):

    count = 1

    def __init__(self, det, wx, wy, vmax,dt=1/30,H=None,H_P=None,H_Q=None,maneuver_time=2,t1=0.9,t2=0.9,window_len=5):
        # xl, vxl, axl, yl, vyl, ayl, h1, h4, h7, h2, h5, h8, h3, h6
        self.alpha = alpha = 1 / maneuver_time

        self.kf = KalmanFilter(dim_x=14, dim_z=2)
        self.motion_transition_mat = np.array([[1, dt, 1 / (alpha**2) * (-1 + alpha * dt + np.e**(-alpha*dt))], 
                                               [0, 1, 1 / alpha * (1 - np.e**(-alpha*dt))], 
                                               [0, 0, np.e**(-alpha*dt)]])
        self.motion_transition_mat = scipy.linalg.block_diag(self.motion_transition_mat, self.motion_transition_mat)

        self.A = np.r_[H, [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)
        self.InvA_orig = self.InvA.copy()

        det_feet, det_R = self.get_UV_and_error(det.get_box())
        x_local, r_local, _ = self.uv2xy(det_feet, det_R, H_P)

        self.R = scipy.linalg.block_diag(r_local, det_R)
        self.prev_r_update = self.R.copy()

        self.kf.x[StateIndex.xl.value] = x_local[0] #xl
        self.kf.x[StateIndex.vxl.value] = 0 #vxl
        self.kf.x[StateIndex.axl.value] = 0 #vxl
        self.kf.x[StateIndex.yl.value] = x_local[1] #yl
        self.kf.x[StateIndex.vyl.value] = 0 #vyl
        self.kf.x[StateIndex.ayl.value] = 0 #vyl
        self.kf.x[-8:, 0] = H

        self.last_xy = self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :]

        self.kf.P = np.zeros((6, 6))
        # np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3, wx ** 2, 1,  vmax**2/3, wy ** 2]))
        np.fill_diagonal(self.kf.P, np.array([1, (vmax / 3)**2, wx ** 2, 1,  (vmax / 3)**2, wy ** 2]))
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

        self.id = SingerKalmanTracker.count
        SingerKalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1

        self.status = TrackStatus.Tentative

        self.window = window_len
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
            [[t1, 1-t1],
             [1-t2, t2]]
        )
        self.N = 2  # number of filters
        self.likelihood = np.zeros(self.N) + 1
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
        self.likelihood[0] = relative_p
        # np.exp(
        #     -0.5 * (np.log(np.linalg.det(2*np.pi*S_0)) + mahala1)
        # )
        self.g_mahala = relative_p
        self.likelihood[1] = relative_iou
        self.relative_iou = relative_iou
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        if np.isnan(self.mu).any():
            self.mu = np.asarray([0.5, 0.5])

        self._compute_mixing_probabilities()

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

        self.prev_r_update = R.copy()
        S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + R[2:4, 2:4]#self.R[2:4, 2:4]
        SI_0 = np.linalg.inv(S_0)

        kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0

        # x = self.kf.x + kalman_gain @ diff[:2]

        # A = np.r_[x[-8:, 0], [1]].reshape((3, 3)).T

        # xy1[:2, :] = x[[StateIndex.xl.value, StateIndex.yl.value], :]

        # b = A @ xy1
        # gamma = 1 / b[2,:]
        # uv_proj = b[:2,:] * gamma

        # residual = y[2:4] - uv_proj

        # # if self.death_count > 1:
        # #     self.residual_list = [residual @ residual.T + (S_0 - self.R[2:4, 2:4])]
        # # else:
        # self.residual_list.append(residual @ residual.T + (S_0 - self.R[2:4, 2:4]))
        # if len(self.residual_list) > self.window:
        #     self.residual_list.pop(0)

        # self.R = np.mean(self.residual_list, axis=0)
        # b = np.dot(self.InvA_orig, np.r_[uv_proj, [[1]]])
        # gamma = 1 / b[2,:]
        # dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du
        # self.R = scipy.linalg.block_diag(dX_dU @ self.R[:2, :2] @ dX_dU.T, self.R)
        # self.R[:2, :2] += np.eye(2) * 1e-12
        # self.R[2:4, 2:4] += np.eye(2)

        # S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + self.R[2:4, 2:4]
        # SI_0 = np.linalg.inv(S_0)
        # kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0
        self.kf.x = self.kf.x + kalman_gain @ diff[:2]
        self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ jacobian[:2]) @ self.kf.P

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        xy1[:2, :] = self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :]
        b = self.A @ xy1
        gamma = 1 / b[2,:]
        self.uv = (b[:2,0] * gamma).astype(int)

        # if self.death_count > 1:
        #     self.innov_list = [kalman_gain @ diff[:2] @ diff[:2].T @ kalman_gain.T]
        # else:
        self.innov_list.append(kalman_gain @ diff[:2] @ diff[:2].T @ kalman_gain.T)
        if len(self.innov_list) > self.window:
            self.innov_list.pop(0)
        self.H_Q = np.mean(self.innov_list, axis=0)[-8:, -8:] + np.eye(8) * 1e-12

        if self.death_count > 1:
            self.box_buffer = [y[4:8].squeeze()]
            self.box_pred = self.box_buffer[-1]
        else:
            self.box_buffer.append(y[4:8].squeeze())
            if len(self.box_buffer) > self.window:
                self.box_buffer.pop(0)

    def compute_mixed_initial(self):
        # compute mixed initial conditions

        # Transform states so that they can be mixed
        last_feet = self.box_buffer[-1].copy()
        last_feet[0] += (last_feet[2] - last_feet[0]) / 2
        last_feet[1] += (last_feet[3] - last_feet[1])
        last_feet = last_feet[:2, None]

        g_pos, sigma_rg, jac = self.uv2xy(last_feet, self.prev_r_update[2:4, 2:4], self.kf.P[-8:, -8:])
        
        # Mix
        omega = self.omega.T
        
        new_x = omega[0, 0] * self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :] + omega[0, 1] * g_pos
        y1, y2 = self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :] - new_x, g_pos - new_x
        new_cov = omega[0, 0] * (np.outer(y1, y1) + self.kf.P[[StateIndex.xl.value, StateIndex.yl.value], :][:, [StateIndex.xl.value, StateIndex.yl.value]]) + omega[0, 1] * (np.outer(y2, y2) + sigma_rg)

        self.kf.x[[StateIndex.xl.value, StateIndex.yl.value], :] = new_x
        self.kf.P[[StateIndex.xl.value, StateIndex.yl.value], StateIndex.xl.value] = new_cov[:, 0]
        self.kf.P[[StateIndex.xl.value, StateIndex.yl.value], StateIndex.yl.value] = new_cov[:, 1]

    def predict(self, affine):
        augmented_affine = np.r_[affine, [[0, 0, 1]]]
        
        if self.death_count == 1:
            if len(self.box_buffer) > 1:
                self.box_pred = self.box_buffer[-1] + self.death_count * np.mean(
                np.array(self.box_buffer)[1:] - np.array(self.box_buffer[:-1]), 
                axis=0)
            self.compute_mixed_initial()
            
        self.kf.F = scipy.linalg.block_diag(
            *(
                self.motion_transition_mat, 
                scipy.linalg.block_diag(
                    np.kron(np.eye(2), augmented_affine),
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

        if self.death_count > 1: # coasted
            wh = affine[:2, :2] @ np.array([[self.w], [self.h]])
            self.w = wh[0,0]
            self.h = wh[1,0]
            self.box_pred = np.array([self.uv[0]-wh[0,0]/2, self.uv[1]-wh[1,0], 
                                    self.uv[0]+wh[0,0]/2, self.uv[1]])
            self.cbar[0] = self.g_mahala = self.cbar[0] * np.exp(-(self.death_count - 1) * self.dt / 4)
            self.cbar[1] = self.relative_iou = 1 - self.cbar[0]
            # self.compute_mixed_initial()

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
        R = np.array(R)[:, :2, :2]
        # R = np.repeat(np.expand_dims(self.R[:2, :2], 0), len(y), 0)

        S1 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + R

        try:
            SI = np.linalg.inv(S1)
            logdet1 = np.linalg.det(np.dot(jacobian[:2, :4], np.dot(self.kf.P[:4, :4],jacobian[:2, :4].T)) + R)
            logdet1 = np.log(logdet1)
        except (RuntimeWarning, LinAlgError):
            return [6000] * len(y), [0] * len(y), [0] * len(y)

        mahalanobis_1 = diff[:, :2].transpose(0, 2, 1) @ SI @ diff[:, :2]
        mahalanobis_1[np.isnan(mahalanobis_1)] = 6000
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

        proba_1 = 1 - scipy.stats.chi2.cdf(mahalanobis_1.squeeze() + logdet1, df=self.kf.dim_x)
        # proba_1 = proba_1 * proba_1 / (proba_1.sum() if proba_1.sum() > 0 else 1)

        proba = (self.cbar[0] * proba_1 * bious + self.cbar[1] * bious) #* np.e ** (-(self.death_count - 1)/5)
        # proba = proba_1 * bious
        # return 1 - proba + max(0, abs(0.5 - self.cbar[0])), bious, proba_1
        return 1 - proba, bious, proba_1

class KalmanTrackerStatic(object):
    def __init__(self, det, wx, wy, vmax,dt=1/30,H=None,H_P=None,H_Q=None, window=5, dynR=False):
        self.window = window
        # xl, vxl, axl, yl, vyl, ayl, h1, h4, h7, h2, h5, h8, h3, h6
        self.kf = KalmanFilter(dim_x=12, dim_z=2)
        self.motion_transition_mat = np.array([[1, dt], 
                                               [0, 1]])
        self.motion_transition_mat = scipy.linalg.block_diag(self.motion_transition_mat, self.motion_transition_mat)

        self.A = np.r_[H, [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)
        self.InvA_orig = self.InvA.copy()

        det_feet, det_R = self.get_UV_and_error(det.get_box())
        x_local, r_local, _ = self.uv2xy(det_feet, det_R, H_P)

        self.R = scipy.linalg.block_diag(r_local, det_R)
        self.kf.x[StateIndexCV.xl.value] = x_local[0] #xl
        self.kf.x[StateIndexCV.vxl.value] = 0 #vxl
        self.kf.x[StateIndexCV.yl.value] = x_local[1] #yl
        self.kf.x[StateIndexCV.vyl.value] = 0 #vyl
        self.kf.x[-8:, 0] = H.copy()

        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, (vmax / 3)**2, 1,  (vmax / 3)**2]))
        self.kf.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], StateIndexCV.xl.value] = (1 * r_local)[:, 0]
        self.kf.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], StateIndexCV.yl.value] = (1 * r_local)[:, 1]

        self.kf.P = scipy.linalg.block_diag(*(self.kf.P, H_P))

        self.x = self.kf.x
        self.P = self.kf.P

        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.Q = np.dot(np.dot(G, Q0), G.T)
        self.H_Q = H_Q
        self.H_Q_orig = self.H_Q.copy() # Used in dynamic filter, do not remove

        self.kf.F = scipy.linalg.block_diag(
            *(
                self.motion_transition_mat, 
                np.eye(8)
            )
        )
        self.kf.Q = scipy.linalg.block_diag(
            self.Q, self.H_Q
        )

        self.dt = dt
        self.likelihood = sys.float_info.min
        self.residual_list = queue.Queue(window)
        self.innov_list = queue.Queue(self.window)
        self.dynR = dynR

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

    def update(self, y):
        y,R = y
        A = self.A

        xy = self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]
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

        jacobian[:2, [StateIndexCV.xl.value, StateIndexCV.yl.value]] = dU_dX
        jacobian[:2, -8:] = dU_dA
        diff = y[2:4] - uv_proj
        S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + (self.R[2:4, 2:4] if self.dynR else R[2:4, 2:4])

        log_likelihood = logpdf(x=diff, cov=S_0)
        self.likelihood = np.exp(log_likelihood)
        if self.likelihood == 0:
            self.likelihood = sys.float_info.min

        SI_0 = np.linalg.inv(S_0)

        kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0

        if self.dynR:
            x = self.kf.x + kalman_gain @ diff[:2]

            A = np.r_[x[-8:, 0], [1]].reshape((3, 3)).T

            xy1[:2, :] = x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]

            b = A @ xy1
            gamma = 1 / b[2,:]
            uv_proj = b[:2,:] * gamma

            residual = y[2:4] - uv_proj

            if self.residual_list.full():
                self.residual_list.get()
            self.residual_list.put(residual @ residual.T + (S_0 - self.R[2:4, 2:4]))

            self.R = np.mean(self.residual_list.queue, axis=0)
            b = np.dot(self.InvA_orig, np.r_[uv_proj, [[1]]])
            gamma = 1 / b[2,:]
            dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du
            self.R = scipy.linalg.block_diag(dX_dU @ self.R[:2, :2] @ dX_dU.T, self.R)
            self.R[:2, :2] += np.eye(2) * 1e-12
            self.R[2:4, 2:4] += np.eye(2)
            # if np.linalg.det(self.R[2:4, 2:4]) > np.linalg.det(R[2:4, 2:4]):
            #     self.R = R

            S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + self.R[2:4, 2:4]
            SI_0 = np.linalg.inv(S_0)
            kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0

        self.x = self.kf.x = self.kf.x + kalman_gain @ diff[:2]
        self.P = self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ jacobian[:2]) @ self.kf.P

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        xy1[:2, :] = self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]
        b = self.A @ xy1
        gamma = 1 / b[2,:]
        self.uv = (b[:2,0] * gamma).astype(int)

        if self.innov_list.full():
            self.innov_list.get()
        self.innov_list.put(kalman_gain @ diff[:2] @ diff[:2].T @ kalman_gain.T)
        self.H_Q = np.mean(self.innov_list.queue, axis=0)[-8:, -8:] + np.eye(8) * 1e-32
        # if np.linalg.det(self.H_Q) > np.linalg.det(self.H_Q_orig):
            # self.H_Q = self.H_Q_orig
            # self.H_Q = self.H_Q * np.linalg.det(self.H_Q_orig) / np.linalg.det(self.H_Q)

    def predict(self, affine):  
        self.kf.x = self.x
        self.kf.P = self.P

        self.kf.Q = scipy.linalg.block_diag(
            self.Q, self.H_Q
        )   
        self.kf.predict()

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        b = np.dot(self.A, np.array([[self.kf.x[StateIndexCV.xl.value, 0]], [self.kf.x[StateIndexCV.yl.value, 0]], [1]]))
        self.uv = (b[:2, 0] / b[-1, 0]).astype(int)  # image proj

        self.x = self.kf.x
        self.P = self.kf.P

    def get_state(self):
        return self.kf.x

    def get_UV_and_error(self, box):
        uv = np.array([[box[0]+box[2]/2], [box[1]+box[3]]])
        u_err,v_err = getUVError(box, Mapper.sigma_m)
        sigma_uv = np.identity(2)
        sigma_uv[0,0] = u_err*u_err
        sigma_uv[1,1] = v_err*v_err
        return uv, sigma_uv
    

class KalmanTrackerDynamic(KalmanTrackerStatic):
    def __init__(self, det, wx, wy, vmax,dt=1/30,H=None,H_P=None,H_Q=None, window=5, dynR=False):
        super().__init__(det, wx, wy, vmax,dt,H,H_P,H_Q, window, dynR)

    def predict(self, affine):
        self.kf.x = self.x
        self.kf.P = self.P

        augmented_affine = np.r_[affine, [[0, 0, 1]]]
            
        self.kf.F = scipy.linalg.block_diag(
            *(
                self.motion_transition_mat, 
                scipy.linalg.block_diag(
                    np.kron(np.eye(2), augmented_affine),
                    affine[:2, :2],
                )
            )
        )

        self.kf.Q = scipy.linalg.block_diag(
            self.Q, self.H_Q
        )
        
        self.kf.predict()
        self.kf.x[-2:, 0] += affine[:2, -1]

        self.x = self.kf.x
        self.P = self.kf.P

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        b = np.dot(self.A, np.array([[self.kf.x[StateIndexCV.xl.value, 0]], [self.kf.x[StateIndexCV.yl.value, 0]], [1]]))
        self.uv = (b[:2, 0] / b[-1, 0]).astype(int)  # image proj

    def update(self, y):
        y,R = y
        A = self.A

        xy = self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]
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

        jacobian[:2, [StateIndexCV.xl.value, StateIndexCV.yl.value]] = dU_dX
        jacobian[:2, -8:] = dU_dA
        diff = y[2:4] - uv_proj

        S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + (self.R[2:4, 2:4] if self.dynR else R[2:4, 2:4])

        log_likelihood = logpdf(x=diff, cov=S_0)
        self.likelihood = np.exp(log_likelihood)
        if self.likelihood == 0:
            self.likelihood = sys.float_info.min

        SI_0 = np.linalg.inv(S_0)

        kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0

        if self.dynR:
            x = self.kf.x + kalman_gain @ diff[:2]

            A = np.r_[x[-8:, 0], [1]].reshape((3, 3)).T

            xy1[:2, :] = x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]

            b = A @ xy1
            gamma = 1 / b[2,:]
            uv_proj = b[:2,:] * gamma

            residual = y[2:4] - uv_proj

            if self.residual_list.full():
                self.residual_list.get()
            self.residual_list.put(residual @ residual.T + (S_0 - self.R[2:4, 2:4]))

            self.R = np.mean(self.residual_list.queue, axis=0)
            b = np.dot(self.InvA_orig, np.r_[uv_proj, [[1]]])
            gamma = 1 / b[2,:]
            dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du
            self.R = scipy.linalg.block_diag(dX_dU @ self.R[:2, :2] @ dX_dU.T, self.R)
            self.R[:2, :2] += np.eye(2) * 1e-12
            self.R[2:4, 2:4] += np.eye(2)
            # if np.linalg.det(self.R[2:4, 2:4]) > np.linalg.det(R[2:4, 2:4]):
            #     self.R = R

            S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + self.R[2:4, 2:4]
            SI_0 = np.linalg.inv(S_0)
            kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0

        self.x = self.kf.x = self.kf.x + kalman_gain @ diff[:2]
        self.P = self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ jacobian[:2]) @ self.kf.P

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        xy1[:2, :] = self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]
        b = self.A @ xy1
        gamma = 1 / b[2,:]
        self.uv = (b[:2,0] * gamma).astype(int)

        if self.innov_list.full():
            self.innov_list.get()
        self.innov_list.put(kalman_gain @ diff[:2] @ diff[:2].T @ kalman_gain.T)
        self.H_Q = np.mean(self.innov_list.queue, axis=0)[-8:, -8:] + self.H_Q_orig
        # if np.linalg.det(self.H_Q) > np.linalg.det(100 * self.H_Q_orig):
        #     self.H_Q = self.H_Q * np.linalg.det(100 * self.H_Q_orig) / np.linalg.det(self.H_Q)

class CVHIMM(IMMEstimator):
    count = 1
    def __init__(self, det, wx, wy, vmax,dt=1/30,H=None,H_P=None,H_Q=None, window=5, t1=1-1e-12, t2=1-1e-12):
        self.groundDist = False
        self.dynR = True
        self.mix = True
        cam_t = 0.9

        super().__init__(
            [
                KalmanTrackerStatic(det, wx, wy, vmax,dt,H,H_P,H_Q, window, self.dynR),
                KalmanTrackerDynamic(det, wx, wy, vmax,dt,H,H_P,H_Q, window, self.dynR)
            ],
            [0.5, 0.5],
            np.array([
                [cam_t, 1-cam_t],
                [1-cam_t, cam_t]
            ])
        )

        A_orig = np.r_[H, [1]].reshape((3, 3)).T
        self.InvA_orig = np.linalg.inv(A_orig)

        norm_homography = self.InvA_orig / self.InvA_orig[-1, -1]
        self.camx, self.camy = (
            norm_homography[0, 1] / norm_homography[2, 1],
            norm_homography[1, 1] / norm_homography[2, 1],
        )

        self.camdist = np.sqrt((self.x[StateIndexCV.xl.value] - self.camx) ** 2 + \
                               (self.x[StateIndexCV.yl.value] - self.camy) ** 2)

        self.id = CVHIMM.count
        CVHIMM.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.status = TrackStatus.Tentative

        self.R = self.filters[0].R

        self.window = window
        box_array = np.array([det.bb_left, det.bb_top, det.bb_left + det.bb_width, det.bb_top + det.bb_height])
        self.box_buffer = [box_array]
        self.box_pred = box_array
        self.uv =  np.array([det.foot_x, det.foot_y]).astype(int)

        # Reference https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/IMM.py#L227
        mu = [0.5, 0.5]  # filter proba: ground, img
        self.boxmu = np.asarray(mu) / np.sum(mu)
        #M[i,j] is the probability of switching from filter j to filter i.
        self.boxM = np.array(
            [[t1, 1 - t1],
             [1 - t2, t2]]
        )
        self.boxN = 2
        self.boxlikelihood = np.zeros(self.boxN) + 1
        self.relative_iou = 1
        self.g_mahala = 1
        self.boxomega = np.zeros((self.boxN, self.boxN))
        self.prev_r_update = self.R.copy()

        self._compute_box_mixing_probabilities()

    def _compute_box_mixing_probabilities(self):
        """
        Compute the mixing probability for each filter.
        """

        self.boxcbar = np.dot(self.boxmu, self.boxM)
        for i in range(self.boxN):
            for j in range(self.boxN):
                self.boxomega[i, j] = (self.boxM[i, j]*self.boxmu[i]) / self.boxcbar[j]
    
    def uv2xy(self, uv, sigma_uv):
        A = np.r_[self.x[-8:, 0], [1]].reshape((3, 3)).T
        InvA = np.linalg.inv(A)
        uv1 = np.zeros((3, 1))
        uv1[:2,:] = uv
        uv1[2,:] = 1
        b = np.dot(InvA, uv1)
        gamma = 1 / b[2,:]
        dX_dU = gamma * InvA[:2, :2] - (gamma**2) * b[:2,:] * InvA[2, :2]  # dX/du
        xy = b[:2,:] * gamma

        gamma2 = 1 / (np.dot(A[-1, :2], xy[:, 0]) + 1)
        dU_dA = gamma2 * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv[0, 0], xy[1, 0], 0, -uv[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv[1, 0], 0, xy[1, 0], -uv[1, 0] * xy[1, 0], 0, 1]
                                  ])  # du/dA
        dX_dA = dX_dU @ dU_dA

        sigma_xy = np.dot(np.dot(dX_dU, sigma_uv[:2, :2]), dX_dU.T) + np.dot(np.dot(dX_dA, self.P[-8:, -8:]), dX_dA.T)
        return xy, sigma_xy
    
    def compute_box_mixed_initial(self):
        # compute mixed initial conditions
        # feet = self.box_buffer[-1].copy().flatten()
        # feet[0] += 0.5 * (feet[2] - feet[0])
        # g_pos, sigma_rg = self.uv2xy(feet[[0, 3], None], self.prev_r_update[2:4, 2:4])
        # g_pos, sigma_rg = self.uv2xy(feet[[0, 3], None], self.R[2:4, 2:4])

        self_xy = np.array([[self.x[StateIndexCV.xl.value, 0]],
                    [self.x[StateIndexCV.yl.value, 0]]])
        # old_cov = self.P.copy()

        # Mix
        omega = self.boxomega.T
        
        # new_x = omega[1, 0] * g_pos + omega[1, 1] * self.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]
        # y1, y2 = self.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :] - new_x, g_pos - new_x
        # new_cov = omega[1, 0] * (np.outer(y2, y2) + sigma_rg) + omega[1, 1] * (np.outer(y1, y1) + self.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], :][:, [StateIndexCV.xl.value, StateIndexCV.yl.value]])

        # self.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :] = new_x
        # self.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], StateIndexCV.xl.value] = new_cov[:, 0]
        # self.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], StateIndexCV.yl.value] = new_cov[:, 1]

        A = np.r_[self.x[-8:, 0], [1]].reshape((3, 3)).T
        b = np.dot(A, np.r_[self_xy, [[1]]])
        uv = b / b[-1, 0]  # image proj
        # gamma = 1 / b[2, :]  # gamma to image
        # dU_dX = gamma * A[:2, :2] - (gamma**2) * b[:2,:] * A[2, :2]


        # dU_dA = gamma * np.array([
        #     [self_xy[0, 0], 0, -self_xy[0, 0] * uv[0, 0], self_xy[1, 0], 0, -uv[0, 0] * self_xy[1, 0], 1, 0],
        #     [0, self_xy[0, 0], -self_xy[0, 0] * uv[1, 0], 0, self_xy[1, 0], -uv[1, 0] * self_xy[1, 0], 0, 1]
        #                           ])  # du/dA

        # jacobian = np.zeros((2, 12))
        # jacobian[:2, [StateIndexCV.xl.value, StateIndexCV.yl.value]] = dU_dX
        # jacobian[:2, -8:] = dU_dA
        # old_cov = np.dot(jacobian[:2], np.dot(old_cov,jacobian[:2].T))

        box_pred = np.array([uv[0]-self.w/2, uv[1]-self.h, 
                             uv[0]+self.w/2, uv[1]])

        self.box_buffer[-1] = omega[0, 0] * self.box_buffer[-1] + omega[0, 1] * box_pred.squeeze()

    def predict(self, affine):
        wh = affine[:2, :2] @ np.array([[self.w], [self.h]])
        self.w = wh[0,0]
        self.h = wh[1,0]
        if self.mix and self.death_count == 1:
            self.compute_box_mixed_initial()

        self.age += 1
        super().predict(affine)

        A = np.r_[self.x[-8:, 0], [1]].reshape((3, 3)).T
        b = np.dot(A, np.array([[self.x[StateIndexCV.xl.value, 0]], [self.x[StateIndexCV.yl.value, 0]], [1]]))
        self.uv = (b[:2, 0] / b[-1, 0]).astype(int)  # image proj

        local_ground_coords = self.InvA_orig @ np.r_[self.uv[:, None], [[1]]]
        local_ground_coords /= local_ground_coords[-1]
        x, y = local_ground_coords[0], local_ground_coords[2]
        self.camdist = np.sqrt((x - self.camx) ** 2 + (y - self.camy) ** 2)

        if len(self.box_buffer) > 1:
            self.box_pred = self.box_buffer[-1] + self.death_count * np.mean(
            np.array(self.box_buffer)[1:] - np.array(self.box_buffer[:-1]), 
            axis=0)
        if self.death_count > 1:
            # self.box_buffer.append(self.box_pred)
            # if len(self.box_buffer) > self.window:
            #     self.box_buffer.pop(0)
            box_pred = np.array([self.uv[0]-wh[0,0]/2, self.uv[1]-wh[1,0], 
                                    self.uv[0]+wh[0,0]/2, self.uv[1]])
            
            self.box_pred = self.boxmu[0] * self.box_pred + self.boxmu[1] * box_pred
        # self.cbar[0] = self.g_mahala = self.cbar[0] * np.exp(-(self.death_count - 1) * self.dt)
        # self.cbar[1] = self.relative_iou = 1 - self.cbar[0]

    def update(self, y, R, relative_iou, g_mahala):
        super().update((y, R))
        self.R = self.mu[0] * self.filters[0].R + self.mu[1] * self.filters[1].R
        if self.death_count > 1:
            # self.box_buffer = [y[4:8].squeeze()]
            self.box_pred = self.box_buffer[-1]
        else:
            self.box_buffer.append(y[4:8].squeeze())
            if len(self.box_buffer) > self.window:
                self.box_buffer.pop(0)

        self.boxlikelihood[0] = max(relative_iou, sys.float_info.min)
        self.relative_iou = relative_iou
        self.boxlikelihood[1] = max(g_mahala, sys.float_info.min)
        self.g_mahala = g_mahala
        self.boxmu = self.boxcbar * self.boxlikelihood
        self.boxmu /= np.sum(self.boxmu)  # normalize
        self._compute_box_mixing_probabilities()
        self.prev_r_update = self.R.copy()

        A = np.r_[self.x[-8:, 0], [1]].reshape((3, 3)).T
        b = np.dot(A, np.array([[self.x[StateIndexCV.xl.value, 0]], [self.x[StateIndexCV.yl.value, 0]], [1]]))
        self.uv = (b[:2, 0] / b[-1, 0]).astype(int)  # image proj

        local_ground_coords = self.InvA_orig @ np.r_[self.uv[:, None], [[1]]]
        local_ground_coords /= local_ground_coords[-1]
        x, y = local_ground_coords[0], local_ground_coords[2]
        self.camdist = np.sqrt((x - self.camx) ** 2 + (y - self.camy) ** 2)

    def distance(self, y, R, buf=0.3):
        A = np.r_[self.x[-8:, 0], [1]].reshape((3, 3)).T
        b = np.dot(A, np.array([[self.x[StateIndexCV.xl.value, 0]], [self.x[StateIndexCV.yl.value, 0]], [1]]))
        uv = b / b[-1, 0]  # image proj
        gamma = 1 / b[2, :]  # gamma to image
        dU_dX = gamma * A[:2, :2] - (gamma**2) * b[:2,:] * A[2, :2]

        self_xy = np.array([[self.x[StateIndexCV.xl.value, 0]],
                            [self.x[StateIndexCV.yl.value, 0]]])
        dU_dA = gamma * np.array([
            [self_xy[0, 0], 0, -self_xy[0, 0] * uv[0, 0], self_xy[1, 0], 0, -uv[0, 0] * self_xy[1, 0], 1, 0],
            [0, self_xy[0, 0], -self_xy[0, 0] * uv[1, 0], 0, self_xy[1, 0], -uv[1, 0] * self_xy[1, 0], 0, 1]
                                  ])  # du/dA
        
        b = self.InvA_orig @ uv  # ground plane coords with H_L
        self_xy = b / b[-1, 0]
        gamma = 1 / b[2, :]
        dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du

        jacobian = np.zeros((2, 12))

        y = np.array(y)
        if self.groundDist:
            jacobian[:2, [StateIndexCV.xl.value, StateIndexCV.yl.value]] = dX_dU @ dU_dX
            jacobian[:2, -8:] = dX_dU @ dU_dA
            diff = (y[:, :2] - self_xy[:2, :])
            if self.dynR:
                R = np.repeat(np.expand_dims(self.R[:2, :2], 0), len(y), 0)
            else:
                R = np.array(R)[:, :2, :2]
        else:
            jacobian[:2, [StateIndexCV.xl.value, StateIndexCV.yl.value]] = dU_dX
            jacobian[:2, -8:] = dU_dA
            diff = (y[:, 2:4] - uv[:2])
            if self.dynR:
                R = np.repeat(np.expand_dims(self.R[2:4, 2:4], 0), len(y), 0)
            else:
                R = np.array(R)[:, 2:4, 2:4]

        S1 = np.dot(jacobian[:2], np.dot(self.P,jacobian[:2].T))
        S1 = S1 + R
        eigenvalues, self.eigenvectors = np.linalg.eig(S1[0])
        self.eigenvalues = np.sqrt(eigenvalues)
        # try:
        SI = np.linalg.inv(S1)
        logdet1 = np.linalg.det(S1)
        self.logdet = logdet1 = np.log(logdet1)
        # except (RuntimeWarning, LinAlgError):
        #     return [6000] * len(y), [0] * len(y), [0] * len(y)

        mahalanobis_1 = diff[:, :2].transpose(0, 2, 1) @ SI @ diff[:, :2]
        mahalanobis_1[np.isnan(mahalanobis_1)] = 6000
        logdet1[np.isnan(logdet1)] = 6000

        buffered_y = np.c_[y[:, 4] - buf * y[:, 6], y[:, 5] - buf * y[:, 7], y[:, 6] + 2*buf * y[:, 6], y[:, 7] + 2*buf * y[:, 7]]
        bious = iou_batch(np.array([
            self.box_pred[0] - buf * self.box_pred[2],
            self.box_pred[1] - buf * self.box_pred[3],
            self.box_pred[2] + 2*buf*self.box_pred[2],
            self.box_pred[3] + 2*buf*self.box_pred[3],
        ])[None, :], buffered_y)

        proba_1 = 1 - scipy.stats.chi2.cdf(mahalanobis_1.squeeze() + logdet1, df=2*self.filters[0].kf.dim_x)
        
        return mahalanobis_1.squeeze() + logdet1, bious, proba_1

class KalmanTrackerBox(object):

    count = 1

    def __init__(self, det, wx, wy, vmax,dt=1/30,H=None,H_P=None,H_Q=None, window=5, t1=0.9, t2=0.9):
        self.window = window
        # xl, vxl, axl, yl, vyl, ayl, h1, h4, h7, h2, h5, h8, h3, h6
        self.kf = KalmanFilter(dim_x=12, dim_z=2)
        self.motion_transition_mat = np.array([[1, dt], 
                                               [0, 1]])
        self.motion_transition_mat = scipy.linalg.block_diag(self.motion_transition_mat, self.motion_transition_mat)

        self.A = np.r_[H, [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)
        self.InvA_orig = self.InvA.copy()

        x_local, r_local = det.y[:2], det.R[:2, :2]

        self.R = scipy.linalg.block_diag(r_local, det.R[2:4, 2:4])
        self.prev_r_update = self.R.copy()
        self.kf.x[StateIndexCV.xl.value] = x_local[0] #xl
        self.kf.x[StateIndexCV.vxl.value] = 0 #vxl
        self.kf.x[StateIndexCV.yl.value] = x_local[1] #yl
        self.kf.x[StateIndexCV.vyl.value] = 0 #vyl
        self.kf.x[-8:, 0] = H.copy()

        self.last_xy = self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]

        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, (vmax / 3)**2, 1,  (vmax / 3)**2]))
        self.kf.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], StateIndexCV.xl.value] = (1 * r_local)[:, 0]
        self.kf.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], StateIndexCV.yl.value] = (1 * r_local)[:, 1]

        self.kf.P = scipy.linalg.block_diag(*(self.kf.P, H_P))

        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.Q = np.dot(np.dot(G, Q0), G.T)
        self.H_Q = H_Q
        self.H_Q_orig = self.H_Q.copy()

        self.id = KalmanTrackerBox.count
        KalmanTrackerBox.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1

        self.status = TrackStatus.Tentative

        self.innov_list = queue.Queue(self.window)
        self.residual_list = queue.Queue(self.window)
        box_array = np.array([det.bb_left, det.bb_top, det.bb_left + det.bb_width, det.bb_top + det.bb_height])
        self.box_buffer = [box_array]
        self.box_pred = box_array
        self.relative_iou = 1
        self.g_mahala = 1

        self.dt = dt
        self.uv =  np.array([det.foot_x, det.foot_y]).astype(int)

        # Reference https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/IMM.py#L227
        mu = [0.5, 0.5]  # filter proba: ground, img
        self.mu = np.asarray(mu) / np.sum(mu)
        #M[i,j] is the probability of switching from filter j to filter i.
        self.M = np.array(
            [[t1, 1-t1],
             [1-t2, t2]]
        )
        self.N = 2  # number of filters
        self.likelihood = np.zeros(self.N) + 1
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

    def update(self, y, R, relative_iou, g_mahala):
        self.likelihood[0] = max(relative_iou, sys.float_info.min)
        # np.exp(
        #     -0.5 * (np.log(np.linalg.det(2*np.pi*S_0)) + mahala1)
        # )
        self.relative_iou = relative_iou
        self.likelihood[1] = max(g_mahala, sys.float_info.min)
        self.g_mahala = g_mahala
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        if np.isnan(self.mu).any():
            self.mu = np.asarray([0.5, 0.5])

        self._compute_mixing_probabilities()

        A = self.A

        xy = self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]
        xy1 = np.ones((3, 1))
        xy1[:2, :] = xy

        b1 = A @ xy1
        gamma1 = 1 / b1[2,:]
        uv_proj1 = b1[:2,:] * gamma1

        jacobian = np.zeros((2, self.kf.dim_x))

        dU_dA = gamma1 * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv_proj1[0, 0], xy[1, 0], 0, -uv_proj1[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv_proj1[1, 0], 0, xy[1, 0], -uv_proj1[1, 0] * xy[1, 0], 0, 1]
                                  ])
        
        dU_dX = gamma1 * A[:2, :2] - (gamma1**2) * b1[:2,:] * A[2, :2]

        jacobian[:2, [StateIndexCV.xl.value, StateIndexCV.yl.value]] = dU_dX
        jacobian[:2, -8:] = dU_dA
        diff = y[2:4] - uv_proj1

        self.prev_r_update = R.copy()
        S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + self.R[2:4, 2:4]
        SI_0 = np.linalg.inv(S_0)

        kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0

        x = self.kf.x + kalman_gain @ diff[:2]

        A = np.r_[x[-8:, 0], [1]].reshape((3, 3)).T

        xy1[:2, :] = x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]

        b = A @ xy1
        gamma = 1 / b[2,:]
        uv_proj = b[:2,:] * gamma

        residual = y[2:4] - uv_proj

        # if self.death_count > 1:
        #     self.residual_list = [residual @ residual.T + (S_0 - self.R[2:4, 2:4])]
        # else:
        # self.residual_list.append(residual @ residual.T + (S_0 - self.R[2:4, 2:4]))
        if self.residual_list.full():
            self.residual_list.get()
        self.residual_list.put(residual @ residual.T + (S_0 - self.R[2:4, 2:4]))
        # if len(self.residual_list) > self.window:
        #     self.residual_list.pop(0)

        self.R = np.mean(self.residual_list.queue, axis=0)
        b = np.dot(self.InvA_orig, np.r_[uv_proj, [[1]]])
        gamma = 1 / b[2,:]
        dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du
        self.R = scipy.linalg.block_diag(dX_dU @ self.R[:2, :2] @ dX_dU.T, self.R)
        # self.R[:2, :2] += np.eye(2) * 1e-12
        # self.R[2:4, 2:4] += np.eye(2)

        S_0 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + self.R[2:4, 2:4]
        SI_0 = np.linalg.inv(S_0)
        kalman_gain = self.kf.P @ jacobian[:2].T @ SI_0
        self.kf.x = self.kf.x + kalman_gain @ diff[:2]
        self.kf.P = (np.eye(self.kf.P.shape[0]) - kalman_gain @ jacobian[:2]) @ self.kf.P

        self.A = np.r_[self.kf.x[-8:, 0], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        xy1[:2, :] = self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :]
        b = self.A @ xy1
        gamma = 1 / b[2,:]
        self.uv = (b[:2,0] * gamma).astype(int)

        # if self.death_count > 1:
        #     self.innov_list = [kalman_gain @ diff[:2] @ diff[:2].T @ kalman_gain.T]
        # else:
        # self.innov_list.append(kalman_gain @ diff[:2] @ diff[:2].T @ kalman_gain.T)
        if self.innov_list.full():
            self.innov_list.get()
        self.innov_list.put(kalman_gain @ diff[:2] @ diff[:2].T @ kalman_gain.T)
        # if len(self.innov_list) > self.window:
        #     self.innov_list.pop(0)
        self.H_Q = np.mean(self.innov_list.queue, axis=0)[-8:, -8:] + self.H_Q_orig#np.eye(8) * 1e-6

        if self.death_count > 1:
            self.box_buffer = [y[4:8].squeeze()]
            self.box_pred = self.box_buffer[-1]
        else:
            self.box_buffer.append(y[4:8].squeeze())
            if len(self.box_buffer) > self.window:
                self.box_buffer.pop(0)
    
    def uv2xy(self, uv, sigma_uv):
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

        sigma_xy = np.dot(np.dot(dX_dU, sigma_uv[:2, :2]), dX_dU.T) + np.dot(np.dot(dX_dA, self.kf.P[-8:, -8:]), dX_dA.T)
        return xy, sigma_xy
    
    def compute_mixed_initial(self):
        # compute mixed initial conditions
        feet = self.box_buffer[-1].copy().flatten()
        feet[0] += 0.5 * (feet[2] - feet[0])
        g_pos, sigma_rg = self.uv2xy(feet[[0, 3], None], self.prev_r_update[2:4, 2:4])

        # Mix
        omega = self.omega.T
        
        new_x = omega[0, 0] * self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :] + omega[0, 1] * g_pos
        y1, y2 = self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :] - new_x, g_pos - new_x
        new_cov = omega[0, 0] * (np.outer(y1, y1) + self.kf.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], :][:, [StateIndexCV.xl.value, StateIndexCV.yl.value]]) + omega[0, 1] * (np.outer(y2, y2) + sigma_rg)

        self.kf.x[[StateIndexCV.xl.value, StateIndexCV.yl.value], :] = new_x
        self.kf.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], StateIndexCV.xl.value] = new_cov[:, 0]
        self.kf.P[[StateIndexCV.xl.value, StateIndexCV.yl.value], StateIndexCV.yl.value] = new_cov[:, 1]

    def predict(self, affine):
        augmented_affine = np.r_[affine, [[0, 0, 1]]]

        if self.death_count == 1:
            if len(self.box_buffer) > 1:
                self.box_pred = self.box_buffer[-1] + self.death_count * np.mean(
                np.array(self.box_buffer)[1:] - np.array(self.box_buffer[:-1]), 
                axis=0)
            self.compute_mixed_initial()
            
        self.kf.F = scipy.linalg.block_diag(
            *(
                self.motion_transition_mat, 
                scipy.linalg.block_diag(
                    np.kron(np.eye(2), augmented_affine),
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

        b = np.dot(self.A, np.array([[self.kf.x[StateIndexCV.xl.value, 0]], [self.kf.x[StateIndexCV.yl.value, 0]], [1]]))
        self.uv = (b[:2, 0] / b[-1, 0]).astype(int)  # image proj

        if self.death_count > 1: # coasted
            wh = affine[:2, :2] @ np.array([[self.w], [self.h]])
            self.w = wh[0,0]
            self.h = wh[1,0]
            self.box_pred = np.array([self.uv[0]-wh[0,0]/2, self.uv[1]-wh[1,0], 
                                    self.uv[0]+wh[0,0]/2, self.uv[1]])
            self.cbar[0] = self.g_mahala = self.cbar[0] * np.exp(-(self.death_count - 1) * self.dt / 4)
            self.cbar[1] = self.relative_iou = 1 - self.cbar[0]
            # self.compute_mixed_initial()

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
        b = np.dot(self.A, np.array([[self.kf.x[StateIndexCV.xl.value, 0]], [self.kf.x[StateIndexCV.yl.value, 0]], [1]]))
        uv = b / b[-1, 0]  # image proj
        gamma = 1 / b[2, :]  # gamma to image
        dU_dX = gamma * self.A[:2, :2] - (gamma**2) * b[:2,:] * self.A[2, :2]

        self_xy = np.array([[self.kf.x[StateIndexCV.xl.value, 0]],
                            [self.kf.x[StateIndexCV.yl.value, 0]]])
        dU_dA = gamma * np.array([
            [self_xy[0, 0], 0, -self_xy[0, 0] * uv[0, 0], self_xy[1, 0], 0, -uv[0, 0] * self_xy[1, 0], 1, 0],
            [0, self_xy[0, 0], -self_xy[0, 0] * uv[1, 0], 0, self_xy[1, 0], -uv[1, 0] * self_xy[1, 0], 0, 1]
                                  ])  # du/dA
        
        b = self.InvA_orig @ uv  # ground plane coords with H_L
        self_xy = b / b[-1, 0]
        gamma = 1 / b[2, :]
        dX_dU = gamma * self.InvA_orig[:2, :2] - (gamma**2) * b[:2,:] * self.InvA_orig[2, :2]  # dX/du

        jacobian = np.zeros((2, self.kf.dim_x))
        jacobian[:2, [StateIndexCV.xl.value, StateIndexCV.yl.value]] = dX_dU @ dU_dX
        jacobian[:2, -8:] = dX_dU @ dU_dA

        y = np.array(y)
        diff = (y[:, :2] - self_xy[:2, :])
        # diff = (y[:, 2:4] - uv[:2])
        # R = np.array(R)[:, :2, :2]
        # R = np.array(R)[:, 2:4, 2:4]
        R = np.repeat(np.expand_dims(self.R[:2, :2], 0), len(y), 0)
        # R = np.repeat(np.expand_dims(self.R[2:4, 2:4], 0), len(y), 0)

        S1 = np.dot(jacobian[:2], np.dot(self.kf.P,jacobian[:2].T)) + R

        # try:
        SI = np.linalg.inv(S1)
        logdet1 = np.linalg.det(S1)
        logdet1 = np.log(logdet1)
        # except (RuntimeWarning, LinAlgError):
        #     return [6000] * len(y), [0] * len(y), [0] * len(y)

        mahalanobis_1 = diff[:, :2].transpose(0, 2, 1) @ SI @ diff[:, :2]
        mahalanobis_1[np.isnan(mahalanobis_1)] = 6000
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

        proba_1 = 1 - scipy.stats.chi2.cdf(mahalanobis_1.squeeze() + logdet1, df=2*self.kf.dim_x)
        # proba_1 = proba_1 * proba_1 / (proba_1.sum() if proba_1.sum() > 0 else 1)

        proba = (self.cbar[0] * proba_1 * bious + self.cbar[1] * bious) #* np.e ** (-(self.death_count - 1)/5)
        # proba = proba_1 * bious
        # return 1 - proba + max(0, abs(0.5 - self.cbar[0])), bious, proba_1
        return 1 - proba_1 * bious, bious, proba_1
        return mahalanobis_1.squeeze() + logdet1, bious, proba_1
    
class OGKalmanTracker(object):

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w,h,dt=1/30):
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.kf.R = R
        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))
        self.kf.P[[0, 2], 0] = (1 * R)[:, 0]
        self.kf.P[[0, 2], 2] = (1 * R)[:, 1]
    
        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.kf.Q = np.dot(np.dot(G, Q0), G.T)

        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = y[1]
        self.kf.x[3] = 0

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.w = w
        self.h = h
        self.mu = self.uv = [0, 0]
        self.g_mahala = 0
        self.relative_iou = 0

        self.status = TrackStatus.Tentative


    def update(self, y, R, _, k_):
        self.kf.update(y[:2, :2],R[:2, :2])

    def predict(self, _):
        self.kf.predict()
        self.age += 1
        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        return self.kf.x
    
    def distance(self, y, R, _):
        diff = np.array(y)[:, :2, :2] - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + np.array(R)[:, :2, :2]
        SI = np.linalg.inv(S)
        mahalanobis = diff.transpose(0, 2, 1) @ SI @ diff
        logdet = np.log(np.linalg.det(S))
        return mahalanobis.squeeze() + logdet, 1, 1
