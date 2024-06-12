import scipy
from .mapper import Mapper
from .gmc import GMCLoader
import numpy as np


# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.foot_x = self.bb_left + self.bb_width / 2
        self.foot_y = self.bb_top + self.bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        # self.y = np.zeros((2, 1))
        # self.R = np.eye(4)

    def get_box(self):
        return [self.bb_left, self.bb_top, self.bb_width, self.bb_height]


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从文本文件读取任意一帧中的目标检测的结果
class Detector:
    def __init__(self, noise_degree=0, frame_width=1920, frame_height=1080, dt=1/30, axis='z'):
        self.seq_length = 0
        self.gmc = None
        self.noise_degree = noise_degree
        self.axis = axis
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.dt = dt

    def load(self,cam_para_file, det_file, gmc_file = None,p_alpha=0,sigma_m=0.05):
        self.mapper = Mapper(cam_para_file,"MOT17",p_alpha,noise_degree=self.noise_degree,frame_width=self.frame_width,frame_height=self.frame_height,dt=self.dt,sigma_m=sigma_m)
        self.load_detfile(det_file)

        if gmc_file is not None:
            self.gmc = GMCLoader(gmc_file)

    def load_detfile(self, filename):
        
        self.dets = dict()
        # 打开文本文件filename
        with open(filename, 'r') as f:
            # 读取文件中的每一行
            for line in f.readlines():
                # 将每一行的内容按照空格分开
                line = line.strip().split(',')
                frame_id = int(line[0])
                if frame_id > self.seq_length:
                    self.seq_length = frame_id
                det_id = int(line[1])
                # 新建一个Detection对象
                det = Detection(det_id)
                det.bb_left = float(line[2])
                det.bb_top = float(line[3])
                det.bb_width = float(line[4])
                det.bb_height = float(line[5])
                det.foot_x = det.bb_left + det.bb_width / 2
                det.foot_y = det.bb_top + det.bb_height
                det.conf = float(line[6])
                det.det_class = int(line[7])
                if det.det_class == -1:
                    det.det_class = 0
                
                if self.noise_degree:
                    if frame_id % 2 == 0:
                        noise_z = self.noise_degree/180.0*np.pi
                    else:
                        noise_z = -self.noise_degree/180.0*np.pi
                    self.mapper.disturb_campara(noise_z, self.axis)

                det.y,det.R = self.mapper.get_UV_and_error(det.get_box())
                y, R = self.mapper.uv2xy(det.y, det.R)
                det.y, det.R = np.r_[y, det.y, [[det.bb_width]]], scipy.linalg.block_diag(R, det.R)
                
                if self.noise_degree:
                    self.mapper.reset_campara()

                # 将det添加到字典中
                if frame_id not in self.dets:
                    self.dets[frame_id] = []
                self.dets[frame_id].append(det)

    def get_dets(self, frame_id,conf_thresh = 0,det_class = 0):
        dets = self.dets[frame_id]
        dets = [det for det in dets if det.det_class == det_class and det.conf >= conf_thresh]

        # for det in dets:
        #     det.y,det.R = self.mapper.get_UV_and_error(det.get_box())
        #     y, R = self.mapper.uv2xy(det.y, det.R)
        #     det.y, det.R = np.r_[y, det.y, [[det.bb_width]]], scipy.linalg.block_diag(R, det.R)
        return dets

    def cmc(self,x,y,w,h,frame_id):
        u,v = self.mapper.xy2uv(x,y)
        affine = self.gmc.get_affine(frame_id)
        M = affine[:,:2]
        T = np.zeros((2,1))
        T[0,0] = affine[0,2]
        T[1,0] = affine[1,2]

        p_center = np.array([[u],[v-h/2]])
        p_wh = np.array([[w],[h]])
        p_center = np.dot(M,p_center) + T
        p_wh = np.dot(M,p_wh)

        u = p_center[0,0]
        v = p_center[1,0]+p_wh[1,0]/2

        xy,_ = self.mapper.uv2xy(np.array([[u],[v]]),np.eye(2))

        return xy[0,0],xy[1,0]


