import itertools
import numpy as np
import os
import scipy
import scipy.linalg


def getUVError(box):
    u, v = 0.05 * box[2], 0.05 * box[3]

    if u>13:
        u = 13
    elif u<2:
        u = 2
    if v>10:
        v = 10
    elif v<2:
        v = 2
    return u,v
    

def parseToMatrix(data, rows, cols):
    matrix_data = np.fromstring(data, sep=' ')
    matrix_data = matrix_data.reshape((rows, cols))
    return matrix_data

def readKittiCalib(filename):
    # 检查文件是否存在
    if not os.path.isfile(filename):
        print(f"Calib file could not be opened: {filename}")
        return None,False

    P2 = np.zeros((3, 4))
    R_rect = np.identity(4)
    Tr_velo_cam = np.identity(4)
    KiKo = None

    with open(filename, 'r') as infile:
        for line in infile:
            id, data = line.split(' ', 1)
            if id == "P2:":
                P2 = parseToMatrix(data, 3, 4)
            elif id == "R_rect":
                R_rect[:3, :3] = parseToMatrix(data, 3, 3)
            elif id == "Tr_velo_cam":
                Tr_velo_cam[:3, :4] = parseToMatrix(data, 3, 4)
            KiKo = np.dot(np.dot(P2, R_rect), Tr_velo_cam)

    return KiKo, True

def readCamParaFile(camera_para):
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    IntrinsicMatrix = np.zeros((3, 3))

    try:
        with open(camera_para, 'r') as f_in:
            lines = f_in.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "RotationMatrices":
                i += 1
                for j in range(3):
                    R[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            elif lines[i].strip() == "TranslationVectors":
                i += 1
                T = np.array(list(map(float, lines[i].split()))).reshape(-1,1)
                T = T / 1000
                i += 1
            elif lines[i].strip() == "IntrinsicMatrix":
                i += 1
                for j in range(3):
                    IntrinsicMatrix[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            else:
                i += 1
    except FileNotFoundError:
        print(f"Error! {camera_para} doesn't exist.")
        return None,False

    Ki = np.zeros((3, 4))
    Ki[:, :3] = IntrinsicMatrix

    Ko = np.eye(4)
    Ko[:3, :3] = R
    Ko[:3, 3] = T.flatten()

    KiKo = np.dot(Ki, Ko)

    return Ki,Ko,True

class Mapper(object):
    def __init__(self, campara_file,dataset= "kitti",process_alpha=0,noise_degree=0,frame_width=1920,frame_height=1080,dt=1/30):
        self.A = np.zeros((3, 3))
        if dataset == "kitti":
            self.KiKo, self.is_ok = readKittiCalib(campara_file)
            self.z0 = -1.73
        else:
            self.Ki,self.Ko, self.is_ok = readCamParaFile(campara_file)
            self.KiKo = np.dot(self.Ki, self.Ko)
            self.z0 = 0

        homogs = []
        # error_degs = np.arange(-0.5, 0.51, 0.1)
        # z0s = np.arange(-1, 1.1, 0.2)
        # prod = itertools.product(error_degs, z0s)
        # for error_deg, z0 in prod:
        #     self.disturb_campara(error_deg, z0)
        #     self.A /= self.A[-1, -1]
        #     homogs.append(self.A.copy().flatten())
        #     self.reset_campara()
        # self.covariance = np.cov(np.array(homogs).T)[:8, :8]

        self.reset_campara()
        if noise_degree > 0:
            self.disturb_campara(noise_degree)

        self.A /= self.A[-1, -1]
        self.InvA = np.linalg.inv(self.A)

        self.process_alpha = process_alpha

        # self.covariance = \
        # np.array([
        #     [22100.95217831667, -471.74454841457754, 3.993376642941903, 12825.089083953644, 2559.2162525084555, -1.9222746253780418, -2665616.4545330876, 118640.42175579324],
        #     [-471.74454841457754, 10.452737142558105, -0.0847176810083616, -274.27919421686386, -54.81140249200246, 0.04081268209462464, 56960.8981261388, -2541.3878089657574],
        #     [3.993376642941903, -0.08471768100836162, 0.0007228795821609905, 2.31679506974895, 0.4621102037490969, -0.0003479636004694695, -481.5971754964347, 21.430775240759523],
        #     [12825.089083953642, -274.27919421686386, 2.31679506974895, 7443.929965221376, 1485.3790803802692, -1.1152170324863155, -1547025.4422417828, 68868.92273824145],
        #     [2559.2162525084555, -54.811402492002465, 0.4621102037490969, 1485.3790803802694, 296.5702279261423, -0.2222987925582746, -308689.7779272262, 13742.050620772834],
        #     [-1.922274625378042, 0.04081268209462464, -0.0003479636004694695, -1.1152170324863155, -0.2222987925582746, 0.0001678344189739308, 231.84843578289627, -10.313850943932446],
        #     [-2665616.4545330876, 56960.8981261388, -481.5971754964347, -1547025.442241783, -308689.77792722615, 231.84843578289627, 321526429.6870994, -14311747.490882616],
        #     [118640.42175579324, -2541.387808965757, 21.430775240759523, 68868.92273824145, 13742.050620772832, -10.313850943932444, -14311747.490882615, 637480.5332748594]
        # ]) 
        # self.covariance /= self.covariance.max()
        self.covariance = np.eye(8) * 1e-12# * (720 / frame_height)**2

        # self.process_covariance = np.array([
        #     [1.691150931731723, 0.027796535337745388, 0.0006057967689015404, 0.7958454550718719, 0.13484592368210732, -0.00032707262341580366, -187.41534165849208, 6.7204375279830275],
        #     [0.027796535337745388, 0.006441797203833354, 1.9769577707441957e-05, 0.009599653979124571, 0.0006343926023067716, -8.26410353986821e-06, -2.6014190135632274, -0.0490543259812558],
        #     [0.0006057967689015404, 1.9769577707441957e-05, 2.469244193298151e-07, 0.00028209410350965624, 4.308977935575974e-05, -1.2944418780092754e-07, -0.06646347291813817, 0.0024582533174924703],
        #     [0.7958454550718719, 0.009599653979124571, 0.00028209410350965624, 0.38863120011711993, 0.06379453575500853, -0.00015322502803559153, -89.79409965684852, 3.3762956539985574],
        #     [0.13484592368210732, 0.0006343926023067716, 4.308977935575974e-05, 0.06379453575500853, 0.012987191990209836, -2.2613232506864336e-05, -14.950530428975313, 0.49244234683260724],
        #     [-0.00032707262341580366, -8.26410353986821e-06, -1.2944418780092754e-07, -0.00015322502803559153, -2.2613232506864336e-05, 7.467887557570941e-08, 0.03636317558003982, -0.0013145955183940965],
        #     [-187.41534165849208, -2.6014190135632274, -0.06646347291813817, -89.79409965684852, -14.950530428975313, 0.03636317558003982, 21001.77798558338, -761.7908047894807],
        #     [6.7204375279830275, -0.0490543259812558, 0.0024582533174924703, 3.3762956539985574, 0.49244234683260724, -0.0013145955183940965, -761.7908047894807, 45.996392809465235]
        # ]) 
        self.process_covariance = self.covariance
        # self.process_covariance = np.eye(8) * 1e-4 * (720 / frame_height)**2 * (0.04 / dt)**2

        # scalar_mat = np.array([
        #     [960 / frame_width, 0, 0],
        #     [0, 540 / frame_height, 0],
        #     [0, 0, 1]
        # ])
        # scalar_block = scipy.linalg.block_diag(scalar_mat, scalar_mat, scalar_mat[:2, :2])
        # self.covariance = scalar_block @ self.covariance @ scalar_block.T

    # def predict(self, affine, trackers):
    def predict(self, affine):
        self.A = np.r_[affine, [[0, 0, 1]]] @ self.A
        assert self.A[-1, -1] == 1
        self.InvA = np.linalg.inv(self.A)
        trans_mat = scipy.linalg.block_diag(
            np.kron(np.eye(2), np.r_[affine, [[0, 0, 1]]]),
            affine[:2, :2],
        )
        self.covariance = trans_mat @ self.covariance @ trans_mat.T + self.process_covariance

    def update(self, tracks, dets):
        if not len(tracks):
            return

        feet_measurements = []
        measurement_covs = []
        track_means = []
        meas_size = 2
        state_size = 4
        jacobian = np.zeros((len(tracks) * meas_size, len(tracks) * state_size + 8))
        temp_cov = np.zeros((len(tracks) * state_size + 8, len(tracks) * state_size + 8))
        uv_projs = []
        for t_idx, track in enumerate(tracks):
            det_idx = track.detidx
            track_means.append(track.kf.x[:4])

            xy1 = np.zeros((3, 1))
            xy = track.kf.x
            xy1[:2, :] = xy[[0, 2]]
            xy1[2, :] = 1
            b = np.dot(self.A, xy1)
            gamma = 1 / b[2,:]

            uv_proj = b[:2,:] * gamma
            uv_projs.append(uv_proj)

            dU_dX = np.zeros((2, 4))
            dU_dX[:, [0, 2]] = gamma * self.A[:2, :2] - (gamma**2) * b[:2,:] * self.A[2, :2]
            jacobian[t_idx * meas_size: (t_idx + 1) * meas_size,
                    t_idx * state_size: (t_idx + 1) * state_size] = dU_dX
            
            dU_dA = gamma * np.array([
                [xy1[0, 0], 0, -xy1[0, 0] * uv_proj[0, 0], xy1[1, 0], 0, -uv_proj[0, 0] * xy1[1, 0], 1, 0],
                [0, xy1[0, 0], -xy1[0, 0] * uv_proj[1, 0], 0, xy1[1, 0], -uv_proj[1, 0] * xy1[1, 0], 0, 1]
                                    ])
            
            jacobian[t_idx * meas_size: (t_idx + 1) * meas_size, -8:] = dU_dA

            temp_cov[t_idx * state_size: (t_idx + 1) * state_size, 
                     t_idx * state_size: (t_idx + 1) * state_size] = track.kf.P[:state_size, :state_size]
            # temp_cov[t_idx * state_size: (t_idx + 1) * state_size, -8:] = track.kf.P[:state_size, -8:]
            # temp_cov[-8:, t_idx * state_size: (t_idx + 1) * state_size] = track.kf.P[-8:, :state_size]

            feet, cov = dets[det_idx].y, dets[det_idx].R
            feet_measurements.append(feet)
            measurement_covs.append(cov)

        temp_cov[-8:, -8:] = self.covariance
        feet_measurements = np.array(feet_measurements)
        measurement_covs = scipy.linalg.block_diag(*measurement_covs)
        
        projected_cov = jacobian @ temp_cov @ jacobian.T + measurement_covs #@ np.diag([100, 100])

        uv_projs = np.array(uv_projs)
        innov = feet_measurements - uv_projs
        inv_projected_cov = np.linalg.inv(projected_cov)

        kalman_gain = temp_cov @ jacobian.T @ inv_projected_cov
        innov = innov.reshape((kalman_gain.shape[1], 1))

        old_means = np.r_[np.array(track_means).reshape((state_size * len(tracks))), self.A.T.flatten()[:-1]] 
        new_mean = old_means + (kalman_gain @ innov.squeeze())
        self.A = np.r_[new_mean[-8:], [1]].reshape((3, 3)).T
        self.InvA = np.linalg.inv(self.A)

        temp_cov = ((np.eye(temp_cov.shape[0]) - kalman_gain @ jacobian) @ temp_cov)
        self.covariance = temp_cov[-8:, -8:]

        self.process_covariance = self.process_alpha * self.process_covariance + ((1 - self.process_alpha) * kalman_gain @ innov @ innov.T @ kalman_gain.T)[-8:, -8:]

    def uv2xy(self, uv, sigma_uv):
        if self.is_ok == False:
            return None, None

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

        sigma_xy = np.dot(np.dot(dX_dU, sigma_uv), dX_dU.T) #+ np.dot(np.dot(dX_dA, self.covariance), dX_dA.T)
        return xy, sigma_xy
    
    def xy2uv(self,x,y):
        if self.is_ok == False:
            return None, None
        xy1 = np.zeros((3, 1))
        xy1[0,0] = x
        xy1[1,0] = y
        xy1[2,0] = 1
        uv1 = np.dot(self.A, xy1)
        return uv1[0,0]/uv1[2,0],uv1[1,0]/uv1[2,0]
    
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
    
    def disturb_campara(self,z, z0):

        # 根据z轴旋转，构造旋转矩阵Rz
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

        R = np.dot(self.Ko[:3, :3],Rz)
        # 将self.Ko 拷贝到新变量 Ko_new
        Ko_new = self.Ko.copy()
        Ko_new[:3, :3] = R
        self.KiKo = np.dot(self.Ki, Ko_new)
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = z0 * self.KiKo[:, 2] + self.KiKo[:, 3]
        self.InvA = np.linalg.inv(self.A)

    def reset_campara(self):
        self.KiKo = np.dot(self.Ki, self.Ko)
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = self.z0 * self.KiKo[:, 2] + self.KiKo[:, 3]

        self.InvA = np.linalg.inv(self.A)


