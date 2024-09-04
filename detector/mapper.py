import numpy as np
import os
import scipy
import scipy.linalg


def getUVError(box, sigma_m=0.05):
    u, v = sigma_m * box[2], sigma_m * box[3]

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
    sigma_m = 0.05
    def __init__(self, campara_file,dataset= "kitti",process_alpha=-29,dt=1/30,sigma_m=0.05):
        self.A = np.zeros((3, 3))
        self.dataset = dataset
        if dataset == "kitti":
            self.KiKo, self.is_ok = readKittiCalib(campara_file)
            self.z0 = -1.73
        else:
            self.Ki,self.Ko, self.is_ok = readCamParaFile(campara_file)
            self.KiKo = np.dot(self.Ki, self.Ko)
            self.z0 = 0

        self.reset_campara()

        self.A /= self.A[-1, -1]
        self.InvA = np.linalg.inv(self.A)

        self.covariance = np.eye(8) * 0#1e-12

        # self.process_covariance = np.eye(8) * 10.0 ** process_alpha #* 1e-3 * ((dt / (1/14))**2) 
        self.process_covariance = np.load("pnoise/dancetrack0007.npy")

        self.sigma_m = sigma_m

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

        gamma2 = 1 / (np.dot(self.A[-1, :2], xy[:, 0]) + 1)
        dU_dA = gamma2 * np.array([
            [xy[0, 0], 0, -xy[0, 0] * uv[0, 0], xy[1, 0], 0, -uv[0, 0] * xy[1, 0], 1, 0],
            [0, xy[0, 0], -xy[0, 0] * uv[1, 0], 0, xy[1, 0], -uv[1, 0] * xy[1, 0], 0, 1]
                                  ])  # du/dA
        dX_dA = dX_dU @ dU_dA

        sigma_xy = np.dot(np.dot(dX_dU, sigma_uv[:2, :2]), dX_dU.T) + np.dot(np.dot(dX_dA, self.covariance), dX_dA.T)
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
        u_err,v_err = getUVError(box, self.sigma_m)
        sigma_uv = np.identity(2)
        sigma_uv[0,0] = u_err*u_err
        sigma_uv[1,1] = v_err*v_err
        return uv, sigma_uv
    
    def disturb_campara(self,z,axis='z'):

        # 根据z轴旋转，构造旋转矩阵Rz
        Rx = np.array([[1,0,0],[0,np.cos(z),-np.sin(z)],[0,np.sin(z),np.cos(z)]])
        Ry = np.array([[np.cos(z),0,np.sin(z)],[0,1,0],[-np.sin(z),0,np.cos(z)]])
        Rz = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])

        R = Rz if axis == 'z' else Ry if axis == 'y' else Rx

        R = np.dot(self.Ko[:3, :3],R)
        # 将self.Ko 拷贝到新变量 Ko_new
        Ko_new = self.Ko.copy()
        Ko_new[:3, :3] = R
        self.KiKo = np.dot(self.Ki, Ko_new)
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = self.z0 * self.KiKo[:, 2] + self.KiKo[:, 3]
        self.InvA = np.linalg.inv(self.A)

    def reset_campara(self):
        if self.dataset != 'kitti':
            self.KiKo = np.dot(self.Ki, self.Ko)
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = self.z0 * self.KiKo[:, 2] + self.KiKo[:, 3]

        self.InvA = np.linalg.inv(self.A)


